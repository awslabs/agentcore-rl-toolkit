"""Download HERB (SalesforceAIResearch/HERB, pinned commit) into a directory: the three code files this example uses
(the agent's system prompt, the structured tools, the evaluation) and the data (30 products, 3 metadata files). Each
file is checked against its sha256.

HERB is CC BY-NC 4.0, and its data was generated with GPT-4o and must not be used to develop models that compete with
OpenAI (upstream README). It is downloaded from its source, not distributed with this example.

    python scripts/fetch_herb.py herb
"""

import hashlib
import io
import sys
import tarfile
from pathlib import Path

import httpx

REPO, COMMIT = "SalesforceAIResearch/HERB", "db3bf9b3f911745726c579c9dbf9f7f6b2c05b36"
FILES = {
    "code/evaluate.py": "f876ddca2ee7d45ac099018487d5f53c33cf763af1ba747459b4135c1a6b9e83",
    "code/metadata_search_functions.py": "1a536dbf12f2e717df40d1fd74d561de388ce92e06969b5023fa9b3fb7c566bd",
    "code/react.py": "352476dbcee9e6edd69cd0e189d5c3e769e41f2d9b057905e973ceb0cd8119e8",
    "data/metadata/customers_data.json": "d17fbfa2d9071000cfcf9a4f9112cd679862f482a3df37b60006448333d3696c",
    "data/metadata/employee.json": "85c7be7ca8f554f27cd84f354dc588aa4abd70ff4e995ca7aa8fd9dd42290747",
    "data/metadata/salesforce_team.json": "549f8aa4f5c4e84dfd28750de212e5d5aadf0751568ae58522796141f0dcb947",
    "data/products/ActionGenie.json": "960f223de786daa74a7d0f70e02cb7f33a457ee9d90c40d8717e35e743b9bf83",
    "data/products/AnomalyForce.json": "a506af89d2a13188ed474f690dab825cabeabb998a3e4136c2f1fee6a95e25a0",
    "data/products/AutoTuneForce.json": "88ec39c0ca02a53e7fe48fb15fcd688d00fc7093374c6be31ca76b7c82823a8c",
    "data/products/CoachForce.json": "e198c4e0e4a9f3b38e446430e18ca7e60d4b2500d00b205e70f8258c7c2cc3b7",
    "data/products/CollaborateForce.json": "4b5c6a20d903a7b89a9cdf2163037d2b4e766fad6fb77f06e5335c0db223ec01",
    "data/products/CollaborationForce.json": "e18eaed9ffbc5210f5be78efefa2d66111b005a690f75cc087826853d45e043b",
    "data/products/ConnectForce.json": "8dbdeea39a6a8a6c32b22a53d03a9f4b3d2dfda5f192d9dccbb91fc60a46315e",
    "data/products/ContentForce.json": "9a0b22d515ea5bc338748b5448bbd07c9f1c37ecb1b141b14a2ccaa12cb7957b",
    "data/products/ContextForce.json": "c46cc6de6faddf3d3b0c8bdb4e6175ec22730b3be9ce483a3eff8789cdd23615",
    "data/products/EdgeForce.json": "3621c068f7e76dafe193c97fd64dfec451678188cca88620d2c8238f60b490c1",
    "data/products/ExplainabilityForce.json": "ffd10bf77d53ddd60862cfa663e5a04f4834d02b0cd824bfce1ab2e837ea798b",
    "data/products/FeedbackForce.json": "f1e41f7a482c118203358106b9cf9b8cee1c492ab64818b75dcdb3d054b95b0b",
    "data/products/FlowForce.json": "be47158645bb696b5de280591a2569550087630a8fedf6efc44433ea105bf935",
    "data/products/ForecastForce.json": "280ec9fc3e4a8573af8fe08f2b2e99ba4dfcf547647ccf55f0fb6038c2bd9805",
    "data/products/InsightForce.json": "15c1a0a6f36d872ee0b1264c18b4bbaca96059857f91f2fbb3258fc4b24074a5",
    "data/products/KnowledgeForce.json": "4b4f90896e35e53b6e0f47f877df444ef80788b95e72bf34fce1588fe89531a7",
    "data/products/LeadForce.json": "6ee1e33bbdce35c1ec4b3e4e95e2eb343f12b1123bafd0da5a225f29b89a699e",
    "data/products/MonitorForce.json": "1ce9aee7300471c664f761de31e1927c3208f0d29c518cc29eea0664e16c5fb6",
    "data/products/PersonalizeForce.json": "de316a8f0e6efc6a44598bedeabda0c8a4b6642da8f06c0569915206bf345754",
    "data/products/PitchForce.json": "041ad14b20da8b6e0070295bc0279b701b0ade23eea13b878a8207d20e1c9142",
    "data/products/ProposalForce.json": "7fbe5d7b84dfc4d0e9acecf908b2db52fa44e5bc0174bd5cedd808d997a45d62",
    "data/products/SearchFlow.json": "1f1a1c2c864ab4771d57bc51ddfcdaaa646db5383b4a03352dae462a579b4b70",
    "data/products/SearchForce.json": "0f2dd4137fca0a38b37d5b3b8099821ac58b78a9ccd4bc92e56a88337faea7a0",
    "data/products/SecurityForce.json": "585431d9a2c654a34f1c99973b642413951588cf40e51450fe56a64764a57ab7",
    "data/products/SentimentForce.json": "9d45b0e9a61c8b99678c205edddc9f521406446421976b5aed1f4a1a85641b71",
    "data/products/SummarizeForce.json": "9fb7096b87eb521b82a3c474eb089ea2a22669c10d755a1bc8373541fd1c925c",
    "data/products/SupportForce.json": "52c40d358ff4aa8c122db18ce1a07da439bbfbf336384a89c563da49d1fba00f",
    "data/products/TrendForce.json": "6f11a2ca897e29ab5007c6863c645cb2a742e258531416d4716fb7e2aea32fcc",
    "data/products/VizForce.json": "96497c45548c03c1ad188a5d759b45aecfaf720efb70d9c6626493f9d194a457",
    "data/products/WorkFlowGenie.json": "290e5d2fa60ec2f978696f7926f2329d21b8b2d427a2ea7d50433de14196aa26",
}


def main(out: str):
    out = Path(out)
    url = f"https://codeload.github.com/{REPO}/tar.gz/{COMMIT}"
    r = httpx.get(url, timeout=300, follow_redirects=True)
    r.raise_for_status()
    found = {}
    with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tar:
        for member in tar.getmembers():
            name = member.name.split("/", 1)[-1]
            if name in FILES and member.isfile():
                found[name] = tar.extractfile(member).read()
    missing = sorted(set(FILES) - set(found))
    if missing:
        sys.exit(f"missing from the archive: {missing}")
    for name, data in sorted(found.items()):
        digest = hashlib.sha256(data).hexdigest()
        if digest != FILES[name]:
            sys.exit(f"sha256 mismatch for {name}: {digest}")
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    print(f"{len(found)} files from {REPO}@{COMMIT[:8]} -> {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "herb")
