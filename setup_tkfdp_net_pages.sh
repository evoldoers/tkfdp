#!/bin/bash
# Cutover tkfdp.net from the S3/CloudFront redirect (managed by
# setup_tkfdp_net_redirect.sh + setup_tkfdp_net_https.sh) to GitHub
# Pages, served from the evoldoers/tkfdp.net repo.
#
# After this runs:
#   apex  tkfdp.net     -> GitHub Pages (4 apex A records)
#   sub   www.tkfdp.net -> evoldoers.github.io  (CNAME)
#
# The old S3 redirect bucket and CloudFront distribution are NOT
# deleted by this script — see the "Tear-down" section below if you
# want to free the (negligible) CloudFront cost.
#
# Run from the AWS account/profile that owns the Route53 hosted zone
# for tkfdp.net (the tkf-gpu sub-account does NOT have Route53
# access; you need the root/main-account creds).
#
# Usage: AWS_PROFILE=<route53-owning-profile> bash setup_tkfdp_net_pages.sh
set -euo pipefail

DOMAIN=tkfdp.net

# GitHub Pages apex A-record targets, per
# https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site/managing-a-custom-domain-for-your-github-pages-site#configuring-an-apex-domain
PAGES_A_RECORDS=(
  185.199.108.153
  185.199.109.153
  185.199.110.153
  185.199.111.153
)
PAGES_CNAME_TARGET=evoldoers.github.io

echo "==> 1. Find Route53 hosted zone ID for $DOMAIN"
ZONE_ID=$(aws route53 list-hosted-zones-by-name --dns-name "$DOMAIN." \
  --query "HostedZones[?Name==\`$DOMAIN.\`].Id | [0]" \
  --output text | sed 's|/hostedzone/||')
if [ -z "$ZONE_ID" ] || [ "$ZONE_ID" = "None" ]; then
  echo "ERROR: no Route53 hosted zone for $DOMAIN found." >&2
  echo "Check your AWS profile has Route53 access." >&2
  exit 1
fi
echo "    hosted zone: $ZONE_ID"

echo "==> 2. Replace the apex A-alias (CloudFront/S3) with 4 plain A records pointing at GitHub Pages"
# Build the ResourceRecords JSON array.
RR_JSON=$(printf '{"Value":"%s"},' "${PAGES_A_RECORDS[@]}" | sed 's/,$//')
cat > /tmp/r53-pages-apex.json <<EOF
{
  "Comment": "Cut tkfdp.net over from S3/CloudFront redirect to GitHub Pages",
  "Changes": [{
    "Action": "UPSERT",
    "ResourceRecordSet": {
      "Name": "$DOMAIN.",
      "Type": "A",
      "TTL": 300,
      "ResourceRecords": [$RR_JSON]
    }
  }]
}
EOF

# If the existing record is an Alias (it is — set by
# setup_tkfdp_net_redirect.sh or setup_tkfdp_net_https.sh), Route53
# rejects an UPSERT that changes the record type/shape. Delete the
# alias first, then create the new plain A records.
echo "==> 2a. Delete the existing apex Alias record (if any)"
EXISTING_ALIAS=$(aws route53 list-resource-record-sets --hosted-zone-id "$ZONE_ID" \
  --query "ResourceRecordSets[?Name==\`$DOMAIN.\` && Type==\`A\` && AliasTarget!=null] | [0]" \
  --output json)
if [ -n "$EXISTING_ALIAS" ] && [ "$EXISTING_ALIAS" != "null" ]; then
  cat > /tmp/r53-pages-delete-alias.json <<EOF
{
  "Comment": "Delete existing apex alias before swap to plain A records",
  "Changes": [{
    "Action": "DELETE",
    "ResourceRecordSet": $EXISTING_ALIAS
  }]
}
EOF
  aws route53 change-resource-record-sets --hosted-zone-id "$ZONE_ID" \
    --change-batch file:///tmp/r53-pages-delete-alias.json
  echo "    deleted existing apex Alias record"
else
  echo "    no existing apex Alias to delete"
fi

echo "==> 2b. Create the 4 GitHub Pages A records"
aws route53 change-resource-record-sets --hosted-zone-id "$ZONE_ID" \
  --change-batch file:///tmp/r53-pages-apex.json

echo "==> 3. Upsert CNAME www.$DOMAIN -> $PAGES_CNAME_TARGET"
cat > /tmp/r53-pages-www.json <<EOF
{
  "Comment": "GitHub Pages www CNAME",
  "Changes": [{
    "Action": "UPSERT",
    "ResourceRecordSet": {
      "Name": "www.$DOMAIN.",
      "Type": "CNAME",
      "TTL": 300,
      "ResourceRecords": [{"Value": "$PAGES_CNAME_TARGET"}]
    }
  }]
}
EOF
aws route53 change-resource-record-sets --hosted-zone-id "$ZONE_ID" \
  --change-batch file:///tmp/r53-pages-www.json

echo
echo "==> Done."
echo "    Wait 5-30 min for DNS to propagate, then test:"
echo "      dig +short $DOMAIN"
echo "      (expect 185.199.108-111.153)"
echo "      curl -I https://$DOMAIN"
echo "      (expect 200 OK from GitHub Pages)"
echo
echo "==> In the evoldoers/tkfdp.net repo settings:"
echo "    1. Settings -> Pages -> Custom domain = $DOMAIN (it should auto-populate from CNAME file)"
echo "    2. Wait for the DNS check to go green"
echo "    3. Tick 'Enforce HTTPS'"
echo
echo "==> Tear-down of the OLD redirect (optional, ~\$0.01/mo savings):"
echo "    aws s3 rb s3://$DOMAIN --force"
echo "    # plus: aws cloudfront delete-distribution (after disabling first)"
echo "    # plus: aws acm delete-certificate (for the old us-east-1 cert)"
