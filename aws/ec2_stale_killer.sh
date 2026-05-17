#!/bin/bash
# Background watchdog: terminate any GPU EC2 instance running for >4 hours.
# Prints + terminates every 15 min.
while true; do
  CUTOFF=$(date -u -d '4 hours ago' +%Y-%m-%dT%H:%M:%SZ)
  STALE=$(AWS_PROFILE=tkf-gpu aws ec2 describe-instances \
    --filters "Name=instance-state-name,Values=running" \
              "Name=instance-type,Values=g5.xlarge,g6.xlarge,g4dn.xlarge,g5.2xlarge,g6.2xlarge,g4dn.2xlarge" \
    --query "Reservations[*].Instances[?LaunchTime<\`$CUTOFF\`].InstanceId" \
    --output text 2>/dev/null)
  if [ -n "$STALE" ]; then
    echo "$(date) STALE GPU EC2 to terminate: $STALE"
    AWS_PROFILE=tkf-gpu aws ec2 terminate-instances --instance-ids $STALE 2>&1 | head -3
  fi
  sleep 900
done
