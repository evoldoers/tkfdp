#!/bin/bash
# Upgrade tkfdp.net redirect from HTTP-only -> HTTPS via ACM + CloudFront.
# Run AFTER setup_tkfdp_net_redirect.sh has created the S3 redirect bucket.
#
# Cost: CloudFront ~$0.01/month for low traffic, ACM cert free.
#
# Usage: AWS_PROFILE=<route53-owning-profile> bash setup_tkfdp_net_https.sh
set -euo pipefail
DOMAIN=tkfdp.net
BUCKET=tkfdp.net
REGION=us-east-1   # ACM cert must be in us-east-1 for CloudFront

echo "==> 1. Request ACM certificate for $DOMAIN (DNS-validated)"
CERT_ARN=$(aws acm request-certificate --domain-name $DOMAIN \
  --validation-method DNS --region $REGION \
  --query CertificateArn --output text)
echo "    cert: $CERT_ARN"

echo "==> 2. Wait 30s for the DNS validation CNAME to populate, then upsert into Route53"
sleep 30
VAL=$(aws acm describe-certificate --certificate-arn "$CERT_ARN" --region $REGION \
  --query 'Certificate.DomainValidationOptions[0].ResourceRecord' --output json)
VNAME=$(echo "$VAL" | python3 -c 'import sys,json; print(json.load(sys.stdin)["Name"])')
VVAL=$(echo "$VAL" | python3 -c 'import sys,json; print(json.load(sys.stdin)["Value"])')
ZONE_ID=$(aws route53 list-hosted-zones-by-name --dns-name $DOMAIN. \
  --query "HostedZones[?Name==\`$DOMAIN.\`].Id | [0]" --output text | sed 's|/hostedzone/||')
cat > /tmp/r53-acm.json <<EOF
{"Changes":[{"Action":"UPSERT","ResourceRecordSet":{
  "Name":"$VNAME","Type":"CNAME","TTL":300,
  "ResourceRecords":[{"Value":"$VVAL"}]}}]}
EOF
aws route53 change-resource-record-sets --hosted-zone-id $ZONE_ID --change-batch file:///tmp/r53-acm.json

echo "==> 3. Wait for cert ISSUED (can be 1-5 min)"
aws acm wait certificate-validated --certificate-arn "$CERT_ARN" --region $REGION
echo "    cert ISSUED"

echo "==> 4. Create CloudFront distribution (S3 website endpoint as custom origin)"
cat > /tmp/cf-dist.json <<EOF
{
  "CallerReference": "tkfdp-$(date +%s)",
  "Aliases": {"Quantity": 1, "Items": ["$DOMAIN"]},
  "DefaultRootObject": "",
  "Origins": {"Quantity": 1, "Items": [{
    "Id": "s3-website-origin",
    "DomainName": "$BUCKET.s3-website-$REGION.amazonaws.com",
    "CustomOriginConfig": {
      "HTTPPort": 80, "HTTPSPort": 443,
      "OriginProtocolPolicy": "http-only",
      "OriginSslProtocols": {"Quantity": 1, "Items": ["TLSv1.2"]},
      "OriginReadTimeout": 30, "OriginKeepaliveTimeout": 5
    }
  }]},
  "DefaultCacheBehavior": {
    "TargetOriginId": "s3-website-origin",
    "ViewerProtocolPolicy": "redirect-to-https",
    "AllowedMethods": {"Quantity": 2, "Items": ["GET","HEAD"],
      "CachedMethods": {"Quantity": 2, "Items": ["GET","HEAD"]}},
    "Compress": true,
    "ForwardedValues": {"QueryString": false,
      "Cookies": {"Forward": "none"},
      "Headers": {"Quantity": 0}},
    "MinTTL": 0, "DefaultTTL": 86400, "MaxTTL": 31536000,
    "TrustedSigners": {"Enabled": false, "Quantity": 0}
  },
  "Comment": "tkfdp.net redirect to github.com/evoldoers/tkfdp",
  "Enabled": true,
  "ViewerCertificate": {
    "ACMCertificateArn": "$CERT_ARN",
    "SSLSupportMethod": "sni-only",
    "MinimumProtocolVersion": "TLSv1.2_2021"
  },
  "PriceClass": "PriceClass_100"
}
EOF
DIST_OUT=$(aws cloudfront create-distribution --distribution-config file:///tmp/cf-dist.json --output json)
DIST_ID=$(echo "$DIST_OUT" | python3 -c 'import sys,json; print(json.load(sys.stdin)["Distribution"]["Id"])')
DIST_DOMAIN=$(echo "$DIST_OUT" | python3 -c 'import sys,json; print(json.load(sys.stdin)["Distribution"]["DomainName"])')
echo "    CloudFront distribution: $DIST_ID -> $DIST_DOMAIN"

echo "==> 5. Replace Route53 A-record alias (S3 website -> CloudFront)"
# CloudFront hosted-zone-id is the fixed AWS constant Z2FDTNDATAQYW2
cat > /tmp/r53-cf.json <<EOF
{"Changes":[{"Action":"UPSERT","ResourceRecordSet":{
  "Name":"$DOMAIN.","Type":"A",
  "AliasTarget":{"HostedZoneId":"Z2FDTNDATAQYW2",
    "DNSName":"$DIST_DOMAIN.","EvaluateTargetHealth":false}}}]}
EOF
aws route53 change-resource-record-sets --hosted-zone-id $ZONE_ID --change-batch file:///tmp/r53-cf.json

echo
echo "==> Done. CloudFront takes ~15-20 min to deploy globally."
echo "    Test: curl -I https://tkfdp.net"
echo "    (expect 301/302 to https://github.com/evoldoers/tkfdp)"
