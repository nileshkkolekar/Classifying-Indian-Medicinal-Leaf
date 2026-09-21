#!/usr/bin/env bash
#
# One-time AWS setup for the CD pipeline.
#
#   AWS_REGION=ap-south-1 BUCKET=leaf-yourname-2026 ./infra/bootstrap-aws.sh
#
# Creates: an ECR repository, a private S3 bucket, the GitHub OIDC provider,
# three IAM roles, CloudWatch log groups, an ECS cluster, and the two task
# definitions. Then prints the GitHub secret and variables to set.
#
# Everything here is idempotent — safe to re-run after a failure.
#
# COST: none of this bills meaningfully on its own. An ECS cluster with no
# tasks is free; S3 and ECR charge for what you store. The bill starts when
# you create *services*, which is a separate step at the end and deliberately
# not automated here, because it needs your VPC details and it is the part
# that runs continuously.

set -euo pipefail

# Git Bash on Windows rewrites any argument that looks like a Unix path into
# a Windows one. CloudWatch log groups are *named* "/ecs/leaf-api" — a name,
# not a path — and would arrive at AWS as "C:/Program Files/Git/ecs/leaf-api",
# which fails validation. Harmless everywhere else; this variable simply does
# not exist on Linux or macOS.
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL="*"

# ── Inputs ───────────────────────────────────────────────────────────────

AWS_REGION="${AWS_REGION:-}"
BUCKET="${BUCKET:-}"

REPO_SLUG="${REPO_SLUG:-nileshkkolekar/Classifying-Indian-Medicinal-Leaf}"
ECR_REPO="${ECR_REPO:-medicinal-leaf}"
CLUSTER="${CLUSTER:-leaf-cluster}"
DEPLOY_ROLE="${DEPLOY_ROLE:-leafGitHubDeployRole}"
EXEC_ROLE="${EXEC_ROLE:-leafEcsExecutionRole}"
TASK_ROLE="${TASK_ROLE:-leafEcsTaskRole}"
# The deploy job declares `environment: production`, which changes the OIDC
# subject claim. Keep these in step or the role will refuse to be assumed.
GH_ENVIRONMENT="${GH_ENVIRONMENT:-production}"

die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
ok()  { printf '  ok    %s\n' "$*"; }
made(){ printf '  made  %s\n' "$*"; }

command -v aws >/dev/null 2>&1 || die "AWS CLI not found. Install it, then run 'aws configure'."
[ -n "$AWS_REGION" ] || die "Set AWS_REGION (e.g. ap-south-1 for Mumbai)."
[ -n "$BUCKET" ] || die "Set BUCKET to a globally unique name (e.g. leaf-<yourname>-2026)."

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)" \
  || die "No valid AWS credentials. Run 'aws configure' first."

say "Account $ACCOUNT_ID · region $AWS_REGION"
echo "  repository: $REPO_SLUG"
echo "  bucket:     $BUCKET"

# ── ECR ──────────────────────────────────────────────────────────────────

say "1/7  Container registry"
if aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$AWS_REGION" >/dev/null 2>&1; then
  ok "ECR repository $ECR_REPO"
else
  aws ecr create-repository \
    --repository-name "$ECR_REPO" \
    --region "$AWS_REGION" \
    --image-scanning-configuration scanOnPush=true \
    --image-tag-mutability MUTABLE >/dev/null
  made "ECR repository $ECR_REPO (scan on push enabled)"
fi

# Untagged layers accumulate on every rebuild and are pure cost.
aws ecr put-lifecycle-policy \
  --repository-name "$ECR_REPO" --region "$AWS_REGION" \
  --lifecycle-policy-text '{
    "rules": [{
      "rulePriority": 1,
      "description": "Expire untagged images after 7 days",
      "selection": {"tagStatus":"untagged","countType":"sinceImagePushed","countUnit":"days","countNumber":7},
      "action": {"type":"expire"}
    }]
  }' >/dev/null
ok "lifecycle policy (untagged images expire after 7 days)"

# ── S3 ───────────────────────────────────────────────────────────────────

say "2/7  Model and dataset bucket"
if aws s3api head-bucket --bucket "$BUCKET" >/dev/null 2>&1; then
  ok "bucket $BUCKET"
else
  if [ "$AWS_REGION" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "$BUCKET" --region "$AWS_REGION" >/dev/null
  else
    aws s3api create-bucket --bucket "$BUCKET" --region "$AWS_REGION" \
      --create-bucket-configuration "LocationConstraint=$AWS_REGION" >/dev/null
  fi
  made "bucket $BUCKET"
fi

aws s3api put-public-access-block --bucket "$BUCKET" \
  --public-access-block-configuration \
  "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true" >/dev/null
ok "public access blocked"

aws s3api put-bucket-encryption --bucket "$BUCKET" \
  --server-side-encryption-configuration \
  '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}' >/dev/null
ok "default encryption on"

# ── OIDC ─────────────────────────────────────────────────────────────────

say "3/7  GitHub OIDC provider"
OIDC_ARN="arn:aws:iam::${ACCOUNT_ID}:oidc-provider/token.actions.githubusercontent.com"
if aws iam get-open-id-connect-provider --open-id-connect-provider-arn "$OIDC_ARN" >/dev/null 2>&1; then
  ok "OIDC provider already present"
else
  # AWS validates GitHub's certificate against its own trust store now, so
  # the thumbprint is required by the API but no longer load-bearing.
  aws iam create-open-id-connect-provider \
    --url https://token.actions.githubusercontent.com \
    --client-id-list sts.amazonaws.com \
    --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1 >/dev/null
  made "OIDC provider — GitHub can now assume roles without stored keys"
fi

# ── IAM ──────────────────────────────────────────────────────────────────

create_role() {  # name, trust-policy-json
  if aws iam get-role --role-name "$1" >/dev/null 2>&1; then
    aws iam update-assume-role-policy --role-name "$1" --policy-document "$2" >/dev/null
    ok "role $1 (trust policy refreshed)"
  else
    aws iam create-role --role-name "$1" --assume-role-policy-document "$2" >/dev/null
    made "role $1"
  fi
}

say "4/7  IAM roles"

create_role "$DEPLOY_ROLE" "$(cat <<JSON
{"Version":"2012-10-17","Statement":[{
  "Effect":"Allow",
  "Principal":{"Federated":"${OIDC_ARN}"},
  "Action":"sts:AssumeRoleWithWebIdentity",
  "Condition":{"StringEquals":{
    "token.actions.githubusercontent.com:aud":"sts.amazonaws.com",
    "token.actions.githubusercontent.com:sub":"repo:${REPO_SLUG}:environment:${GH_ENVIRONMENT}"
  }}
}]}
JSON
)"

ECS_TRUST='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
create_role "$EXEC_ROLE" "$ECS_TRUST"
create_role "$TASK_ROLE" "$ECS_TRUST"

aws iam attach-role-policy --role-name "$EXEC_ROLE" \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy >/dev/null
ok "execution role can pull from ECR and write logs"

# Deploy permissions: scoped to this repository and these two services.
# iam:PassRole is the one that matters — unscoped, anyone able to register a
# task definition could run a container as any role in the account.
aws iam put-role-policy --role-name "$DEPLOY_ROLE" --policy-name leafDeploy \
  --policy-document "$(cat <<JSON
{"Version":"2012-10-17","Statement":[
 {"Sid":"EcrAuthIsAccountWide","Effect":"Allow","Action":"ecr:GetAuthorizationToken","Resource":"*"},
 {"Sid":"PushOnlyToThisRepository","Effect":"Allow",
  "Action":["ecr:BatchCheckLayerAvailability","ecr:CompleteLayerUpload","ecr:InitiateLayerUpload",
            "ecr:PutImage","ecr:UploadLayerPart","ecr:BatchGetImage"],
  "Resource":"arn:aws:ecr:${AWS_REGION}:${ACCOUNT_ID}:repository/${ECR_REPO}"},
 {"Sid":"RegisterTaskDefinitions","Effect":"Allow",
  "Action":["ecs:RegisterTaskDefinition","ecs:DescribeTaskDefinition"],"Resource":"*"},
 {"Sid":"UpdateOnlyTheseServices","Effect":"Allow",
  "Action":["ecs:UpdateService","ecs:DescribeServices"],
  "Resource":["arn:aws:ecs:${AWS_REGION}:${ACCOUNT_ID}:service/${CLUSTER}/leaf-api",
              "arn:aws:ecs:${AWS_REGION}:${ACCOUNT_ID}:service/${CLUSTER}/leaf-ui"]},
 {"Sid":"PassOnlyTheTaskRoles","Effect":"Allow","Action":"iam:PassRole",
  "Resource":["arn:aws:iam::${ACCOUNT_ID}:role/${EXEC_ROLE}",
              "arn:aws:iam::${ACCOUNT_ID}:role/${TASK_ROLE}"],
  "Condition":{"StringEquals":{"iam:PassedToService":"ecs-tasks.amazonaws.com"}}}
]}
JSON
)" >/dev/null
ok "deploy role scoped to this repo, these services, those two roles"

# The application only ever reads. It never needs to write to S3.
aws iam put-role-policy --role-name "$TASK_ROLE" --policy-name leafReadModel \
  --policy-document "$(cat <<JSON
{"Version":"2012-10-17","Statement":[
 {"Sid":"ReadModelAndDataset","Effect":"Allow","Action":"s3:GetObject",
  "Resource":["arn:aws:s3:::${BUCKET}/models/*","arn:aws:s3:::${BUCKET}/datasets/*"]},
 {"Sid":"ListOnlyThosePrefixes","Effect":"Allow","Action":"s3:ListBucket",
  "Resource":"arn:aws:s3:::${BUCKET}",
  "Condition":{"StringLike":{"s3:prefix":["models/*","datasets/*"]}}}
]}
JSON
)" >/dev/null
ok "task role can read models/ and datasets/ — read-only, nothing else"

# ── Logs and cluster ─────────────────────────────────────────────────────

say "5/7  CloudWatch log groups"
for group in /ecs/leaf-api /ecs/leaf-ui; do
  if aws logs describe-log-groups --log-group-name-prefix "$group" --region "$AWS_REGION" \
       --query "logGroups[?logGroupName=='$group']" --output text | grep -q .; then
    ok "$group"
  else
    aws logs create-log-group --log-group-name "$group" --region "$AWS_REGION" >/dev/null
    made "$group"
  fi
  # Logs are cheap but not free, and nobody reads month-old container logs.
  aws logs put-retention-policy --log-group-name "$group" --retention-in-days 14 --region "$AWS_REGION" >/dev/null
done
ok "14-day retention"

say "6/7  ECS cluster"
if aws ecs describe-clusters --clusters "$CLUSTER" --region "$AWS_REGION" \
     --query "clusters[?status=='ACTIVE']" --output text | grep -q .; then
  ok "cluster $CLUSTER"
else
  aws ecs create-cluster --cluster-name "$CLUSTER" --region "$AWS_REGION" >/dev/null
  made "cluster $CLUSTER (free until tasks run)"
fi

# ── Task definitions ─────────────────────────────────────────────────────

say "7/7  Task definitions"
# A directory inside the project, referenced relatively. An absolute path
# from mktemp would be a Unix path like /tmp/tmp.abc, which the Windows
# aws.exe cannot open — the same mismatch, one layer down.
WORK=".aws-render"
rm -rf "$WORK"; mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT

for pair in "ecs-task-definition.json:leaf-api" "ecs-task-definition-ui.json:leaf-ui"; do
  file="${pair%%:*}"; family="${pair##*:}"
  sed -e "s|ACCOUNT_ID|${ACCOUNT_ID}|g" \
      -e "s|AWS_REGION|${AWS_REGION}|g" \
      -e "s|BUCKET_NAME|${BUCKET}|g" \
      "infra/${file}" > "${WORK}/${file}"
  aws ecs register-task-definition --cli-input-json "file://${WORK}/${file}" \
      --region "$AWS_REGION" >/dev/null
  made "registered $family"
done

# ── What to do next ──────────────────────────────────────────────────────

cat <<SUMMARY

────────────────────────────────────────────────────────────────────────
Done. Now set these in GitHub
  Settings → Secrets and variables → Actions

  Secret
    AWS_DEPLOY_ROLE_ARN = arn:aws:iam::${ACCOUNT_ID}:role/${DEPLOY_ROLE}

  Variables
    AWS_REGION      = ${AWS_REGION}
    ECR_REPOSITORY  = ${ECR_REPO}
    ECS_CLUSTER     = ${CLUSTER}
    ECS_API_SERVICE = leaf-api
    ECS_UI_SERVICE  = leaf-ui

  Also create an Environment named "${GH_ENVIRONMENT}" (Settings →
  Environments). The deploy job declares it, and the role's trust policy
  expects that exact subject claim.

Upload the model — the image deliberately contains none:

  aws s3 cp artifacts/checkpoints/best.pt s3://${BUCKET}/models/best.pt
  aws s3 cp artifacts/checkpoints/best.meta.json s3://${BUCKET}/models/best.meta.json

Set authentication credentials on the task definition before the service
runs, or the container starts and rejects every request:

  leaf-hash              # MLC_AUTH__USERS and MLC_AUTH__SECRET_KEY

Then create the services. Left manual on purpose: it needs your VPC
details, and it is the step that starts billing.

  aws ecs create-service --cluster ${CLUSTER} --service-name leaf-api \\
    --task-definition leaf-api --desired-count 1 --launch-type FARGATE \\
    --region ${AWS_REGION} \\
    --network-configuration "awsvpcConfiguration={subnets=[SUBNET_ID],securityGroups=[SG_ID],assignPublicIp=ENABLED}"

Keep desired-count at 1: the job queue is in-process, so a second replica
would answer 404 for jobs submitted to the first.

Set a budget before you walk away:
  https://console.aws.amazon.com/billing/home#/budgets
────────────────────────────────────────────────────────────────────────
SUMMARY
