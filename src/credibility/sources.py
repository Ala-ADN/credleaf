"""Curated institutional sources for the COVID-era comparison corpus.

Expand coverage by reviewing actual unmatched corpus hosts, checking their
institutional ownership, and adding the narrowest suitable hostname below.
Parent domains cover their subdomains; never whitelist a public suffix, an
entire university, a social platform or a generic hosting service.

This is a source-level classification, not a guarantee about every article.
Preprint repositories remain included for compatibility; preprints are not
peer reviewed. Historical COVID portals refer to their pandemic-era operation.

After editing, refresh the saved registry without fetching Archive-It seeds:
    uv run python scripts/build_credibility.py --refresh-authoritative
Then rerun the EDA notebook; it resolves cached georef labels on load.
"""
from __future__ import annotations

# Health authorities (national + supranational). Prefer health-specific domains
# over adding broad government roots. Existing gov.uk / canada.ca entries are
# retained for compatibility with the original study's source definition.
HEALTH_AUTHORITIES = frozenset({
    "who.int",
    "cdc.gov",
    "nih.gov",
    "fda.gov",
    "ecdc.europa.eu",
    "ema.europa.eu",
    "phe.gov.uk",
    "gov.uk",
    "canada.ca",
    "health.gov.au",
    "pasteur.fr",
    "rki.de",
    "iss.it",
    "paho.org",                       # Pan American Health Organization
    "africacdc.org",                  # African Union public health agency
    "unicef.org",                     # UN Children's Fund
    "hhs.gov",                        # US Department of Health and Human Services
    "samhsa.gov",                     # US mental health / substance-use agency
    "medicare.gov",                   # US federal health insurance programme
    "medlineplus.gov",                # US National Library of Medicine
    "coronavirus.gov",                # Historical US government COVID portal
    "covid.gov",                      # Historical US government COVID portal
    "nhs.uk",                         # UK National Health Service
    "nhsinform.scot",                 # NHS Scotland health information
    "publichealthscotland.scot",
    "hse.ie",                         # Ireland's Health Service Executive
    "hpsc.ie",                        # Ireland's Health Protection Surveillance Centre
    "santepubliquefrance.fr",
    "bag.admin.ch",                   # Swiss Federal Office of Public Health
    "foph-coronavirus.ch",            # Historical Swiss FOPH COVID campaign
    "rivm.nl",                        # Dutch public health institute
    "folkhalsomyndigheten.se",        # Public Health Agency of Sweden
    "publichealthontario.ca",
    "bccdc.ca",                       # British Columbia Centre for Disease Control
    "inspq.qc.ca",                    # Quebec public health institute
    "health.govt.nz",                 # New Zealand Ministry of Health
    "covid19.govt.nz",                # Historical NZ government COVID response
    "healthdirect.gov.au",            # Australian government-funded health information
    "chinacdc.cn",                    # Includes weekly.chinacdc.cn
    "mohfw.gov.in",                   # India's Ministry of Health and Family Welfare
    "moh.gov.sg",                     # Singapore Ministry of Health
    "moh.gov.bn",                     # Brunei Ministry of Health
    "doh.gov.ph",                     # Philippines Department of Health
    "moph.gov.qa",                    # Qatar Ministry of Public Health
    "moh.gov.jo",                     # Jordan Ministry of Health
    "ncdc.gov.ng",                    # Nigeria Centre for Disease Control
    "nicd.ac.za",                     # South African communicable-disease institute
    "health.go.ug",                   # Uganda Ministry of Health
    # Institutional references:
    # https://www.nhs.uk/about-us/
    # https://www.health.govt.nz/
    # https://en.chinacdc.cn/research/journals/sponsor/weekly/index.html
    # https://africacdc.org/about-us/
})

# State health departments and official COVID portals, anchored on reviewed
# corpus hosts and including legacy/newer aliases. Broader state/county
# government roots stay outside this list.
# Directory: https://www.cdc.gov/infrastructure/contact/index.html
REGIONAL_HEALTH_AUTHORITIES = frozenset({
    "cdph.ca.gov",                    # California
    "dph.georgia.gov",
    "floridahealth.gov",
    "floridahealthcovid19.gov",        # Historical Florida COVID portal
    "health.ny.gov",                  # New York State
    "dshs.state.tx.us",               # Historical Texas DSHS hostname
    "dshs.texas.gov",
    "vdh.virginia.gov",
    "health.wyo.gov",
    "healthy.arkansas.gov",
    "health.mo.gov",
    "health.state.mn.us",
    "health.nd.gov",
    "dphhs.mt.gov",
    "dph.illinois.gov",
    "doh.sd.gov",
    "nmhealth.org",                   # Includes cv.nmhealth.org
    "health.pa.gov",
    "alabamapublichealth.gov",
    "doh.wa.gov",
    "covid19.colorado.gov",
    "coronavirus.dc.gov",
    "coronavirus.idaho.gov",
    "coronavirus.utah.gov",
    "nvhealthresponse.nv.gov",
    "cchealth.org",                   # Contra Costa Health
    "publichealth.lacounty.gov",
})

# Medical journals and systematic-review organisations.
MEDICAL_JOURNALS = frozenset({
    "nejm.org",
    "thelancet.com",
    "bmj.com",
    "jamanetwork.com",
    "annals.org",
    "cell.com",
    "plos.org",
    "ploscompbiol.org",
    "ploscollections.org",
    "acpjournals.org",                # Annals of Internal Medicine's newer host
    "jmir.org",
    "jci.org",
    "eurosurveillance.org",           # ECDC's infectious-disease journal
    "cochrane.org",
    "cochranelibrary.com",
    "mayoclinicproceedings.org",
})

# Named medical/public-health institutions, using specific departmental hosts
# where the parent university also hosts unrelated or user-generated content.
MEDICAL_INSTITUTIONS = frozenset({
    "cidrap.umn.edu",                 # University of Minnesota infectious-disease centre
    "globalhealthnow.org",            # Johns Hopkins Bloomberg School public-health news
    "publichealth.jhu.edu",
    "jhsph.edu",                      # Historical Bloomberg School hostname
    "coronavirus.jhu.edu",
    "systems.jhu.edu",                # Johns Hopkins CSSE COVID data project
    "hopkinsmedicine.org",
    "healthdata.org",                 # University of Washington IHME
    "health.harvard.edu",             # Harvard Medical School health publishing
    "hms.harvard.edu",
    "hsph.harvard.edu",
    "mayoclinic.org",
    "clevelandclinic.org",
    "lshtm.ac.uk",                    # London School of Hygiene & Tropical Medicine
    # Ownership/editorial references:
    # https://www.cidrap.umn.edu/
    # https://globalhealthnow.org/about
    # https://www.health.harvard.edu/about-us
    # https://www.hopkinsmedicine.org/about
})

# Scientific publishing/repositories. A repository host is not an assertion
# that an individual manuscript has passed peer review.
SCIENCE_PUBLISHERS = frozenset({
    "nature.com",
    "science.org",
    "sciencemag.org",
    "ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
    "biorxiv.org",
    "medrxiv.org",
    "europepmc.org",
})

AUTHORITATIVE_SOURCE_GROUPS: dict[str, frozenset[str]] = {
    "health_authorities": HEALTH_AUTHORITIES,
    "regional_health_authorities": REGIONAL_HEALTH_AUTHORITIES,
    "medical_journals": MEDICAL_JOURNALS,
    "medical_institutions": MEDICAL_INSTITUTIONS,
    "science_publishers": SCIENCE_PUBLISHERS,
}

AUTHORITATIVE_DOMAINS: frozenset[str] = frozenset().union(*AUTHORITATIVE_SOURCE_GROUPS.values())
