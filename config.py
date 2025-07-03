feature_meanings = {
    "Q1": "Gender",
    "Q2": "Age",
    "Q3": "Age Groups",
    "Q4": "Region",
    "Q5": "Category usage",
    "Q6": "Personal Income",
    "Q9": "Usage",
    "Q10": "Usage detail",
    "Q14": "Number of changes",
    "Q15": "Change from PKV to GKV",
    "Q18": "Switchers l10y",

    # Moments of Truth – Key experiences with GKV
    "Q24.1": "Received information about services",
    "Q24.2": "Interaction with service hotline",
    "Q24.3": "Usage of online customer portal",
    "Q24.4": "Filing a reimbursement claim",
    "Q24.5": "Health check-up or preventive care",
    "Q24.6": "Hospital or medical treatment support",
    "Q24.7": "Consultation on supplementary insurance",
    "Q24.8": "Changing tariff or service plan",

    # Brand Perception – Attributes associated with GKV brand
    "Q40.1": "Trustworthiness",
    "Q40.2": "Customer-oriented service",
    "Q40.3": "Transparent communication",
    "Q40.4": "Innovative services",
    "Q40.5": "Good value for money",
    "Q40.6": "Reliable performance",
    "Q40.7": "Ease of access to services",
    "Q40.8": "Modern digital tools",
    "Q40.9": "Support during difficult situations",
    "Q40.10": "Comprehensive health coverage",
    "Q40.11": "Fairness in dealings",
    "Q40.12": "Quick claim processing",
    "Q40.13": "Helpful customer service",
    "Q40.14": "Strong market reputation",


    # Loyalty & NPS
    "Q41": "GKV Loyalty",
    "Q42": "NPS",

    # Pricing
    "Q58": "Additional contribution – Price change levels",

    # Contact Experience
    "Q72": "Last Contact with GKV",
    "Q74": "Contact evaluation GKV",
    "Q75": "Positive contact GKV",
    "Q76": "Negative contact GKV",

    # Zusatzversicherung – Supplemental Insurance Topics
    "Q77.1": "Interest in dental supplemental insurance",
    "Q77.2": "Interest in hospital stay insurance",
    "Q77.3": "Interest in alternative medicine coverage",
    "Q77.4": "Interest in travel insurance",
    "Q77.5": "Interest in daily sickness allowance",
    "Q77.6": "Interest in vision/glasses insurance",
    "Q77.7": "Interest in private room during hospital stays",
    "Q77.8": "Interest in preventive care supplements",
    "Q77.9": "Interest in other supplemental insurance types",

    # Health Role – Personal attitudes and behaviors related to health
    "Q93.1": "Takes responsibility for personal health",
    "Q93.2": "Proactive about preventive care",
    "Q93.3": "Informed about health topics",
    "Q93.4": "Maintains a healthy lifestyle",
    "Q93.5": "Acts quickly on health issues",
    "Q93.6": "Feels in control of personal health",
    "Q93.7": "Seeks professional advice regularly",
    "Q93.8": "Uses digital tools for health tracking",
    "Q93.9": "Encourages others to be health-conscious",
    "Q93.10": "Sees health as a lifelong commitment",

    # Medical Questions – Experiences and assessments of medical services
    "Q94.1": "Satisfaction with general practitioner",
    "Q94.2": "Satisfaction with specialists",
    "Q94.3": "Access to medical appointments",
    "Q94.4": "Waiting time for treatment",
    "Q94.5": "Availability of telemedicine services",
    "Q94.6": "Clarity of medical explanations",
    "Q94.7": "Level of empathy from doctors",
    "Q94.8": "Coordination among different doctors",
    "Q94.9": "Trust in medical professionals",
    "Q94.10": "Ease of getting prescriptions",
    "Q94.11": "Clarity of diagnosis communication",
    "Q94.12": "Confidence in treatment decisions",
    "Q94.13": "Ease of contacting medical staff",
    "Q94.14": "Satisfaction with emergency care",
    "Q94.15": "Access to mental health services",
    "Q94.16": "Use of electronic health records",
    "Q94.17": "Experience with home care",
    "Q94.18": "Coverage for medical expenses",
    "Q94.19": "Comfort during medical treatments",
    "Q94.20": "Follow-up after treatments",

    # Additional Medical Questions
    "Q95.1": "Awareness of covered services",
    "Q95.2": "Use of second opinion services",
    "Q95.3": "Confidence in medical system",
    "Q95.4": "Satisfaction with pediatric care",
    "Q95.5": "Support for chronic illness management",
    "Q95.6": "Availability of rehabilitation services",


    # Household & Background
    "Q98": "Household size",
    "Q99": "Children in household",
    "Q100": "Cultural/ethnic background",
    "Q101": "Detailed cultural/ethnic background",
    "Q102": "Employment status",
    "Q103": "Professional experience",
    "Q104": "Marital status"
}

provider_map = {
    "AOK": 1,
    "IKK classic": 2,
    "IKK gesund plus": 3,
    "IKK Südwest": 4,
    "IKK - Die Innovationskasse": 5,
    "IKK Brandenburg und Berlin": 6,
    "BIG - direkt gesund": 7,
    "DAK Gesundheit": 8,
    "HEK": 9,
    "hkk": 10,
    "SBK": 11,
    "Techniker-Krankenkasse (TK)": 12,
    "Kaufmännische Krankenkasse (KKH)": 13,
    "Knappschaft": 14,
    "Betriebskrankenkasse Mobil": 16,
    "mhplus BKK": 17,
    "pronova BKK": 18,
    "Audi BKK": 19,
    "BAHN-BKK": 20,
    "BARMER": 21,
    "BKK Verkehrsbau Union (BKK VBU)": 22,
    "VIACTIV Krankenkasse": 23
}

# Cluster names and descriptions
cluster_info = {
        0: {
            "name": "Active but Dissatisfied Switchers",
            "description": "Younger members with multiple past switches who had negative insurer experiences, moderate loyalty, and actively evaluate recent contacts. Often responsive to price and service changes."
        },
        1: {
            "name": "Disillusioned Frequent Switchers",
            "description": "Extremely frequent switchers, frustrated by previous private insurance or repeated bad experiences, showing poor loyalty and low satisfaction."
        },
        2: {
            "name": "Information-Seeking Skeptics",
            "description": "Moderate switchers who focus on negative experiences and seek better communication, wanting improved friendliness from insurers."
        },
        3: {
            "name": "Skeptical Active Switchers",
            "description": "Active switchers with a skeptical view of marketing claims, moderate loyalty, and a history of bad experiences driving their choices."
        }
    }
