#!/usr/bin/env bash
#
# Stop paying for the deployment.
#
#   AWS_REGION=ap-south-1 ./infra/teardown-aws.sh            # scale to zero
#   AWS_REGION=ap-south-1 ./infra/teardown-aws.sh --delete   # remove services
#   AWS_REGION=ap-south-1 BUCKET=... ./infra/teardown-aws.sh --purge
#
# Default is to scale services to zero, which stops essentially all the cost
# and is reversible in one command. Fargate bills per vCPU-second while tasks
# run, so scaling down is usually what you actually want between demos.
#
# --delete removes the services and cluster but keeps the image, the bucket
# and the IAM roles, so a redeploy is quick.
#
# --purge additionally deletes the ECR repository and empties the bucket.
# That destroys your uploaded model. It asks first.

set -euo pipefail

AWS_REGION="${AWS_REGION:-}"
BUCKET="${BUCKET:-}"
CLUSTER="${CLUSTER:-leaf-cluster}"
ECR_REPO="${ECR_REPO:-medicinal-leaf}"
SERVICES=("leaf-api" "leaf-ui")

MODE="scale"
case "${1:-}" in
  --delete) MODE="delete" ;;
  --purge)  MODE="purge" ;;
  "")       MODE="scale" ;;
  *)        echo "Unknown option: $1" >&2; exit 2 ;;
esac

die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

command -v aws >/dev/null 2>&1 || die "AWS CLI not found."
[ -n "$AWS_REGION" ] || die "Set AWS_REGION."
aws sts get-caller-identity >/dev/null 2>&1 || die "No valid AWS credentials."

service_exists() {
  aws ecs describe-services --cluster "$CLUSTER" --services "$1" --region "$AWS_REGION" \
    --query "services[?status=='ACTIVE']" --output text 2>/dev/null | grep -q .
}

# ── Always: stop the running tasks ───────────────────────────────────────

say "Scaling services to zero"
for service in "${SERVICES[@]}"; do
  if service_exists "$service"; then
    aws ecs update-service --cluster "$CLUSTER" --service "$service" \
      --desired-count 0 --region "$AWS_REGION" >/dev/null
    echo "  stopped  $service"
  else
    echo "  absent   $service"
  fi
done
echo
echo "  Fargate charges stop once the tasks drain (a minute or so)."
echo "  Bring it back with: aws ecs update-service --cluster $CLUSTER \\"
echo "    --service leaf-api --desired-count 1 --region $AWS_REGION"

[ "$MODE" = "scale" ] && exit 0

# ── --delete: remove services and cluster ────────────────────────────────

say "Deleting services and cluster"
for service in "${SERVICES[@]}"; do
  if service_exists "$service"; then
    aws ecs delete-service --cluster "$CLUSTER" --service "$service" \
      --force --region "$AWS_REGION" >/dev/null
    echo "  deleted  $service"
  fi
done

aws ecs delete-cluster --cluster "$CLUSTER" --region "$AWS_REGION" >/dev/null 2>&1 \
  && echo "  deleted  cluster $CLUSTER" \
  || echo "  cluster $CLUSTER not deleted (it may still be draining; re-run shortly)"

echo
echo "  The image, bucket and IAM roles are intact, so redeploying is quick."

[ "$MODE" = "delete" ] && exit 0

# ── --purge: destroy the registry and the model ──────────────────────────

say "PURGE — this deletes your uploaded model and every built image"
echo "  ECR repository: $ECR_REPO"
echo "  S3 bucket:      ${BUCKET:-(not set, will be skipped)}"
echo
printf "  Type 'purge' to confirm: "
read -r reply
[ "$reply" = "purge" ] || die "Not confirmed; nothing further was removed."

aws ecr delete-repository --repository-name "$ECR_REPO" --force --region "$AWS_REGION" >/dev/null 2>&1 \
  && echo "  deleted  ECR repository $ECR_REPO" \
  || echo "  ECR repository $ECR_REPO already gone"

if [ -n "$BUCKET" ]; then
  # Emptied rather than deleted: the name stays reserved to you, and a
  # bucket name you have used is hard to get back once released.
  aws s3 rm "s3://${BUCKET}" --recursive >/dev/null 2>&1 \
    && echo "  emptied  s3://${BUCKET} (bucket kept, name still yours)" \
    || echo "  could not empty s3://${BUCKET}"
fi

echo
echo "  IAM roles and the OIDC provider were left alone — they cost nothing"
echo "  and re-creating them is the fiddliest part of the setup."
