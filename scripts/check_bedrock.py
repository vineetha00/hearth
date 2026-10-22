"""Check that your AWS setup can call Bedrock before you run Hearth on it.

    python scripts/check_bedrock.py            # test the model in HEARTH_BEDROCK_MODEL
    python scripts/check_bedrock.py --list     # list Anthropic model / inference-profile IDs you can use

Reads AWS_REGION, HEARTH_BEDROCK_MODEL and normal AWS credentials (env vars, ~/.aws, or SSO). Never prints secrets.
"""

from __future__ import annotations

import os
import sys


def hint(exc: Exception) -> str:
    code = getattr(exc, "response", {}).get("Error", {}).get("Code", type(exc).__name__)
    msg = str(exc)
    tips = {
        "AccessDeniedException": "Your credentials work but aren't allowed to call this model. In the Bedrock console, enable access to the model "
                                 "(Model access / Model catalog) and make sure your IAM user/role has Bedrock permissions "
                                 "(e.g. AmazonBedrockFullAccess for a hackathon).",
        "ValidationException": "The model ID or request was rejected. Run with --list and copy an exact ID (often an inference profile "
                               "starting with 'us.'), then set HEARTH_BEDROCK_MODEL. Also check the model is offered in your AWS_REGION.",
        "ResourceNotFoundException": "Model not found in this region. Check AWS_REGION and run with --list.",
        "ThrottlingException": "Rate limited. Wait a moment and retry.",
        "ExpiredTokenException": "Your credentials expired. Re-run `aws sso login` or refresh your keys.",
        "UnrecognizedClientException": "AWS didn't accept the access key. Re-check AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY.",
        "InvalidSignatureException": "The secret key doesn't match the access key. Re-check both.",
    }
    return f"{code}: {tips.get(code, msg[:300])}"


def main() -> int:
    try:
        import boto3
    except ImportError:
        print("boto3 isn't installed. Run: pip install -r requirements.txt")
        return 1

    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-west-2"
    session = boto3.Session(region_name=region)
    creds = session.get_credentials()
    if creds is None:
        print("No AWS credentials found.\n"
              "Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (and AWS_REGION), or run `aws configure`, or `aws sso login` with AWS_PROFILE set.")
        return 1
    print(f"Credentials found (key id ends in ...{creds.access_key[-4:]}), region {region}")

    if "--list" in sys.argv:
        try:
            bedrock = session.client("bedrock")
            print("\nAnthropic inference profiles (use these IDs; they start with a region prefix):")
            for p in bedrock.list_inference_profiles(maxResults=200).get("inferenceProfileSummaries", []):
                if "anthropic" in p["inferenceProfileId"]:
                    print("  ", p["inferenceProfileId"])
            print("\nAnthropic foundation models:")
            for m in bedrock.list_foundation_models(byProvider="anthropic").get("modelSummaries", []):
                print("  ", m["modelId"])
        except Exception as exc:  # noqa: BLE001
            print("Couldn't list models:", hint(exc))
            return 1
        return 0

    model = os.environ.get("HEARTH_BEDROCK_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
    print(f"Calling model: {model}")
    try:
        client = session.client("bedrock-runtime")
        resp = client.converse(
            modelId=model,
            messages=[{"role": "user", "content": [{"text": "Reply with exactly: Hearth is ready."}]}],
            inferenceConfig={"maxTokens": 30, "temperature": 0},
        )
        text = "".join(b.get("text", "") for b in resp["output"]["message"]["content"])
        usage = resp.get("usage", {})
        print(f"\nOK. Model said: {text!r}  (tokens in/out: {usage.get('inputTokens')}/{usage.get('outputTokens')})")
        print("Bedrock works. Set HEARTH_BEDROCK_MODEL to this ID in BOTH terminals, then start the server and simulator.")
        return 0
    except Exception as exc:  # noqa: BLE001
        print("\nFAILED:", hint(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
