#!/usr/bin/env python3
"""Scrape biography (lífshlaup) XML and extract education/career with keyword tags."""

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set
import xml.etree.ElementTree as ET

import requests
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.manual_models import MemberBiography
from app.models import ThingmannalistiThingmadur, Base

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Icelandic keyword patterns for categorization
# Education classification following Statistics Iceland standards
# Level 1: Grunnmenntun (Primary/Basic Education) - no keywords (indicated by absence)
# Level 2: Starfs- og framhaldsmenntun (Vocational and Secondary Education)
# Level 3: Háskólamenntun (Higher/University Education) - organized by field of study

EDUCATION_KEYWORDS = {
    # STARFS- OG FRAMHALDSMENNTUN (Vocational and Secondary Education)
    "menntaskólanám": [
        "stúdentspróf",
        "framhaldsskóla",
        "menntaskóla",
        "fjölbraut",
        "gagnfræðapróf",
        "verslunarpróf",
    ],
    "starfsmenntun": [
        "iðnnám",
        "starfsnám",
        "iðn",
        "handverks",
        "smíð",
        "rafvirkj",
        "steinnám",
        "múrara",
        "prófskírteini",
        "samvinnuskólapróf",
        "fiskimannapróf",
        "stýrimannapróf",
        "rekstrarfræð",
        "stjórnsýslu",
        "búfræðing",
        "búfræðipróf",
    ],

    # HÁSKÓLAMENNTUN (Higher/University Education) - by field of study
    "háskólamenntun: lögfræði": [
        "lögfræð",
        "cand",
        "júrís",
        "réttarfræð",
    ],
    "háskólamenntun: læknisfræði": [
        "lækn",
        "hjúkrun",
        "lyfjafræð",
        "heilsufræð",
        "dýralækn",
    ],
    "háskólamenntun: verkfræði": [
        "verkfræð",
        "tækni",
        "iðnfræð",
        "verkfræðingur",
    ],
    "háskólamenntun: hagfræði": [
        "hagfræð",
        "fjármál",
        "bókhald",
        "þjóðhagfræð",
        "viðskiptafræð",
    ],
    "háskólamenntun: kennsla": [
        "kennar",
        "prófessor",
        "dósent",
        "lektor",
        "kennslufræð",
    ],
    "háskólamenntun: guðfræði": [
        "guðfræð",
        "prestaskól",
        "prestum",
        "deilufræð",
    ],
    "háskólamenntun: raunvísindi": [
        "raunvísind",
        "eðlisfræð",
        "efnafræð",
        "stærðfræð",
        "eðli",
        "líffræð",
        "jarðfræð",
        "tölvunarfræðingur",
        "íþróttafræ",
    ],
    "háskólamenntun: listir": [
        "listir",
        "bókmennt",
        "myndlist",
        "tónlist",
        "leiklist",
        "hönnun",
        "íslens",
    ],
    "háskólamenntun: félagsvísindi": [
        "stjórnmálafræð",
        "félagsfræð",
        "sálfræð",
        "þjóðkynnis",
        "heimspeki",
        "heilbrigðis",
        "lýðheilsu",
        "félagsmál",
        "tómstund",
    ],
    "háskólamenntun: landbúnaður": [
        "landbúnaður",
        "búvísind",
        "bújarðarfræð",
    ],
}

CAREER_KEYWORDS = {
    # Statistics Iceland occupation categories (ISCO-88)
    "stjórnendur": [
        "forstjór", "framkvæmdastjór", "ráðherra", "hluthafi", "eigand", "framsal",
        "verksmiðju", "skrifstofu", "stofnun", "yfirlögmaður",
    ],
    "sérfræðingar": [
        "lögmaður", "hæstaréttarlögmaður", "dómari", "saksóknari", "réttarfærandi", "lögfræðingur",
        "kennar", "prófessor", "dósent", "lektor", "kennslumadur",
        "lækn", "hjúkrun", "heilsugæzlu",
        "blaðamaður", "ritstjór", "útgáva", "blaðamenn",
        "lögreglumaður", "yfirlögregluþjónn", "lögreglustjór", "ríkislögreglustjóra",
        "prestur", "prófastur",
    ],
    "tæknar og sérmenntað starfsfólk": [
        "sjómaður", "fiskimaður", "fiskveiðu",
    ],
    "þjónustu-, sölu- og afgreiðslufólk": [
        "kaupmaður", "kaupmanna", "verslunar", "verslun",
        "flutningsmaður", "flugþjónn", "bílstjóri", "þjónustufulltrúi",
    ],
    "iðnaðarmenn og sérhæft iðnverkafólk": [
        "steinhögg", "múrari", "smiðir", "trésmið",
    ],
    # Special historical categories
    "þingmaður": ["þingmaður", "alþingismaður"],
    "sveitarstjórn": ["bæjar", "hreppstjór", "sveitarstjór"],
    "búskapur": ["bóndi", "búi", "bú\\."],
    "menning": ["leikari", "listamadur", "músík", "safnvörður"],
}


def fetch_biography_xml(url: str, timeout: int = 15) -> Optional[ET.Element]:
    """Fetch and parse biography XML."""
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        return root
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None


def extract_text(element: Optional[ET.Element]) -> str:
    """Extract text from an XML element, cleaning up whitespace."""
    if element is None or element.text is None:
        return ""
    text = element.text.strip()
    # Unescape common HTML entities
    text = text.replace("&mdash;", "—")
    text = text.replace("&ndash;", "–")
    text = text.replace("&nbsp;", " ")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&amp;", "&")
    return text


def apply_tags(text: str, keyword_dict: Dict[str, List[str]]) -> List[str]:
    """Apply keyword-based tags to text."""
    if not text:
        return []
    text_lower = text.lower()
    tags = []
    for tag, keywords in keyword_dict.items():
        for keyword in keywords:
            if re.search(r"\b" + keyword, text_lower):
                tags.append(tag)
                break  # Tag matched, move to next tag
    return sorted(set(tags))  # Unique, sorted


def scrape_biographies(db_url: str, force: bool = False) -> None:
    """Scrape all member biographies from XML and store in database."""
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        # Get all unique members
        members = session.execute(
            select(
                ThingmannalistiThingmadur.attr_id,
                ThingmannalistiThingmadur.leaf_nafn,
                ThingmannalistiThingmadur.leaf_xml_lifshlaup,
            )
            .distinct(ThingmannalistiThingmadur.attr_id)
            .order_by(ThingmannalistiThingmadur.attr_id)
        ).all()

        logger.info(f"Processing {len(members)} unique members")

        for idx, (member_id, member_name, bio_url) in enumerate(members, 1):
            if not bio_url:
                logger.debug(f"[{idx}/{len(members)}] {member_name}: no biography URL")
                continue

            logger.info(f"[{idx}/{len(members)}] Fetching {member_name} (ID: {member_id})")

            # Check if already scraped (unless force)
            existing = session.execute(
                select(MemberBiography).where(MemberBiography.member_id == member_id)
            ).scalar_one_or_none()

            if existing and not force:
                logger.debug(f"  Already scraped, skipping (use --force to re-scrape)")
                continue

            # Fetch biography XML
            root = fetch_biography_xml(bio_url)
            if root is None:
                logger.warning(f"  Failed to fetch biography")
                continue

            # Navigate to the lífshlaup section (it's nested in the root)
            bio_section = root.find("lífshlaup")
            if bio_section is None:
                logger.warning(f"  No <lífshlaup> section found")
                continue

            # Extract education and career text
            education_elem = bio_section.find("menntun")
            career_elem = bio_section.find("starfsferill")

            education_text = extract_text(education_elem)
            career_text = extract_text(career_elem)

            # Apply keyword tagging
            education_tags = apply_tags(education_text, EDUCATION_KEYWORDS)
            career_tags = apply_tags(career_text, CAREER_KEYWORDS)

            # Upsert into database
            if existing:
                existing.education_text = education_text
                existing.career_text = career_text
                existing.education_tags = json.dumps(education_tags)
                existing.career_tags = json.dumps(career_tags)
                existing.source_url = bio_url
                existing.scraped_at = datetime.utcnow().isoformat()
                logger.info(f"  Updated biography")
            else:
                bio = MemberBiography(
                    member_id=member_id,
                    education_text=education_text,
                    career_text=career_text,
                    education_tags=json.dumps(education_tags),
                    career_tags=json.dumps(career_tags),
                    source_url=bio_url,
                    scraped_at=datetime.utcnow().isoformat(),
                )
                session.add(bio)
                logger.info(f"  Inserted biography")

        session.commit()
        logger.info("Done")


def retag_biographies(db_url: str) -> None:
    """Re-apply tags to existing biography text without fetching XML."""
    engine = create_engine(db_url)

    with Session(engine) as session:
        # Get all biographies with text
        bios = session.execute(
            select(MemberBiography).where(
                (MemberBiography.education_text != "") | (MemberBiography.career_text != "")
            )
        ).scalars().all()

        logger.info(f"Retagging {len(bios)} biographies")

        for idx, bio in enumerate(bios, 1):
            if idx % 100 == 0:
                logger.info(f"[{idx}/{len(bios)}] Retagging...")

            education_tags = apply_tags(bio.education_text or "", EDUCATION_KEYWORDS)
            career_tags = apply_tags(bio.career_text or "", CAREER_KEYWORDS)

            bio.education_tags = json.dumps(education_tags)
            bio.career_tags = json.dumps(career_tags)
            session.add(bio)

        session.commit()
        logger.info("Done retagging")


def main():
    parser = argparse.ArgumentParser(
        description="Scrape member biographies from Althingi XML endpoints"
    )
    parser.add_argument(
        "--db",
        default="sqlite:///data/althingi.db",
        help="Database URL (default: sqlite:///data/althingi.db)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-scrape even if already in database",
    )
    parser.add_argument(
        "--retag-only",
        action="store_true",
        help="Re-apply tags to existing text without fetching XML (fast keyword updates)",
    )
    args = parser.parse_args()

    if args.retag_only:
        retag_biographies(args.db)
    else:
        scrape_biographies(args.db, force=args.force)


if __name__ == "__main__":
    main()
