#!/bin/bash
# Set up tkfdp.net -> https://github.com/evoldoers/tkfdp redirect via
# S3 static website + Route53 alias. HTTP-only; for HTTPS see Step 5.
#
# Run from the AWS account that owns the Route53 hosted zone for
# tkfdp.net. The tkf-gpu sub-account does NOT have Route53 access;
# you'll need root/main-account creds or a profile with route53:* + s3:*.
#
# Usage: AWS_PROFILE=<profile_with_route53> bash setup_tkfdp_net_redirect.sh
set -euo pipefail

BUCKET=tkfdp.net
REGION=us-east-1
TARGET_URL_HOST=github.com
TARGET_URL_PATH=evoldoers/tkfdp
# Hosted-zone IDs for S3 website endpoints (AWS-fixed; us-east-1):
S3_WEBSITE_ZONE_ID=Z3AQBSTGFYJSTF

echo "==> 1. Find the hosted zone ID for tkfdp.net"
ZONE_ID=$(aws route53 list-hosted-zones-by-name --dns-name tkfdp.net. \
  --query 'HostedZones[?Name==`tkfdp.net.`].Id | [0]' --output text | sed 's|/hostedzone/||')
if [ -z "$ZONE_ID" ] || [ "$ZONE_ID" = "None" ]; then
  echo "ERROR: no Route53 hosted zone for tkfdp.net found." >&2
  echo "Create one first or check your AWS profile." >&2
  exit 1
fi
echo "    hosted zone: $ZONE_ID"

echo "==> 2. Create S3 bucket '$BUCKET' (idempotent)"
aws s3api create-bucket --bucket $BUCKET --region $REGION 2>&1 \
  | grep -v "BucketAlreadyOwnedByYou" || true

echo "==> 3. Disable Block-Public-Access (S3 website requires public read)"
aws s3api put-public-access-block --bucket $BUCKET \
  --public-access-block-configuration \
    "BlockPublicAcls=false,IgnorePublicAcls=false,BlockPublicPolicy=false,RestrictPublicBuckets=false"

echo "==> 4. Configure bucket as static website with redirect to https://$TARGET_URL_HOST/$TARGET_URL_PATH"
# RoutingRule with KeyPrefixEquals='' matches all paths; ReplaceKeyWith pins the
# target path so any request to tkfdp.net/* redirects to .../evoldoers/tkfdp
cat > /tmp/website-config.json <<EOF
{
  "IndexDocument": {"Suffix": "index.html"},
  "RoutingRules": [{
    "Condition": {"KeyPrefixEquals": ""},
    "Redirect": {
      "Protocol": "https",
      "HostName": "$TARGET_URL_HOST",
      "ReplaceKeyWith": "$TARGET_URL_PATH",
      "HttpRedirectCode": "302"
    }
  }]
}
EOF
aws s3api put-bucket-website --bucket $BUCKET \
  --website-configuration file:///tmp/website-config.json

# Public-read policy (required for the website endpoint to serve the redirect)
cat > /tmp/bucket-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "PublicReadGetObject",
    "Effect": "Allow",
    "Principal": "*",
    "Action": "s3:GetObject",
    "Resource": "arn:aws:s3:::$BUCKET/*"
  }]
}
EOF
aws s3api put-bucket-policy --bucket $BUCKET \
  --policy file:///tmp/bucket-policy.json

S3_WEBSITE_ENDPOINT="$BUCKET.s3-website-$REGION.amazonaws.com"
echo "    S3 website endpoint: http://$S3_WEBSITE_ENDPOINT"

echo "==> 5. Upsert Route53 A-record (alias -> S3 website endpoint)"
cat > /tmp/r53-change.json <<EOF
{
  "Comment": "Alias tkfdp.net -> S3 website redirect to github.com/$TARGET_URL_PATH",
  "Changes": [{
    "Action": "UPSERT",
    "ResourceRecordSet": {
      "Name": "tkfdp.net.",
      "Type": "A",
      "AliasTarget": {
        "HostedZoneId": "$S3_WEBSITE_ZONE_ID",
        "DNSName": "s3-website-$REGION.amazonaws.com.",
        "EvaluateTargetHealth": false
      }
    }
  }]
}
EOF
aws route53 change-resource-record-sets --hosted-zone-id "$ZONE_ID" \
  --change-batch file:///tmp/r53-change.json

echo
echo "==> Done."
echo "    Test in ~5 min once DNS propagates:"
echo "      curl -I http://tkfdp.net"
echo "      (expect '301 Moved Permanently' / '302 Found' to https://github.com/$TARGET_URL_PATH)"
echo
echo "==> NOTE: HTTP-only. For HTTPS support:"
echo "    1. aws acm request-certificate --domain-name tkfdp.net --validation-method DNS"
echo "       (in us-east-1 — required for CloudFront)"
echo "    2. Wait + validate via Route53 (aws acm describe-certificate ... -> CNAME records)"
echo "    3. Create CloudFront distribution: Origin = $S3_WEBSITE_ENDPOINT (custom origin, HTTP only)"
echo "       Alternate domain name = tkfdp.net; SSL cert = the ACM cert from step 1"
echo "    4. Replace the Route53 alias above with one pointing at the CloudFront distribution"
