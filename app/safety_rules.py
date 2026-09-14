"""
Deterministic safety + triage layer.

Design principle: deterministic red flags always have veto power. A separate
structured LLM triage classifier may ESCALATE unseen wording, but it can never
downgrade a deterministic emergency. This module also validates generated output.
"""
import re
from enum import Enum
from dataclasses import dataclass


class Tier(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    URGENT = "urgent"
    SELF_HARM = "self_harm"  # handled completely separately from normal flow


@dataclass
class TriageResult:
    tier: Tier
    matched_rule: str | None = None


# --- Hinglish / Romanized Hindi red-flag patterns ---
# Hindi written using Latin/Roman script is extremely common in WhatsApp
# messages. Keep these deterministic and biased toward recall.

URGENT_PATTERNS_HINGLISH = [
    # Chest pain
    r"\bchest\s+(mein|me|ka)\s+(dard|pain)\b",
    r"\bseene\s+(mein|me|ka)\s+(dard|pain)\b",
    r"\bseena\s+(mein|me)\s+(dard|pain)\b",

    # Breathing difficulty
    # Roman Hindi spelling is highly variable on WhatsApp: saans/saas/sans,
    # nahi/nai/ni, aa raha/a rha, etc. Bias these red flags toward recall.
    r"\b(saa?ns?|saas)\s+(nahi|nahin|nai|ni)\s+(aa|a|aati|ati)(?:\s+(raha|rahi|rha|rhi|hai|hain))?\b",
    r"\b(saa?ns?|saas)\s+(aa|a)\s+(nahi|nahin|nai|ni)\s+(raha|rahi|rha|rhi)\b",
    r"\b(saa?ns?|saas)\s+l[ei]ne\s+(mein|me)\s+(dikkat|problem|pareshani|taklif|takleef)\b",
    r"\b(saa?ns?|saas)\s+(nahi|nahin|nai|ni)\s+le\s+(paa|pa|pa raha|pa rha)\b",
    r"\b(breath|breathing)\s+(nahi|nahin|nai|ni)\s+(aa|a)(?:\s+(raha|rahi|rha|rhi))?\b",
    r"\b(dam|dum)\s+ghut\s+(raha|rahi|rha|rhi)\b",

    # Unconscious / fainting
    r"\bbehosh\b",
    r"\bbehoshi\b",
    r"\bbehosh\s+ho\s+gaya\b",
    r"\bbehosh\s+ho\s+gayi\b",
    r"\bchakkar\s+aa\s+kar\s+gir\b",

    # Stroke / paralysis
    r"\bstroke\b",
    r"\blakva\b",
    r"\bparalysis\b",
    r"\bbody\s+ka\s+ek\s+side\s+(sunn|sun|paraly[sz]ed)\b",

    # Severe bleeding
    r"\bbahut\s+(zyada|zyaada|jyada)\s+khoon\b",
    r"\bkhun\s+(bah|nikal)\s+(raha|rhi|ri)\b",
    r"\bkhoon\s+(bah|nikal)\s+(raha|rhi|ri)\b",
    r"\bsevere\s+bleeding\b",

    # Seizure / fits
    r"\bdaura\s+pad\b",
    r"\bfit\s+pad\b",
    r"\bseizure\b",

    # Severe abdominal pain
    r"\bpet\s+mein\s+(bahut|bohot|bohut)\s+(tez\s+)?dard\b",
    r"\bpet\s+me\s+(bahut|bohot|bohut)\s+(tez\s+)?dard\b",

    # Coughing blood
    r"\bkhansi\s+mein\s+khoon\b",
    r"\bkhoon\s+ki\s+khansi\b",

    # Blue lips
    r"\bhonth\s+(neele|blue)\b",

    # Severe allergic reaction
    r"\bsevere\s+allergic\s+reaction\b",
    r"\banaphylaxis\b",

    # Pregnancy + bleeding/severe pain
    r"\bpregnan\w*\b.*\b(bleeding|khoon|dard)\b",
]

HIGH_RISK_MEDICINE_PATTERNS = [
    # English
    r"\bchild\b",
    r"\bkid\b",
    r"\bbaby\b",
    r"\binfant\b",
    r"\btoddler\b",
    r"\bunder\s+18\b",
    r"\bpregnant\b",
    r"\bpregnancy\b",
    r"\bbreastfeeding\b",
    r"\bbreast\s*feeding\b",
    r"\bliver\s+(disease|problem|condition)\b",
    r"\bkidney\s+(disease|problem|condition)\b",
    r"\brenal\s+(disease|problem|condition)\b",
    r"\bdrug\s+allergy\b",
    r"\bmedicine\s+allergy\b",
    r"\ballergic\s+to\s+(medicine|medication|drug)\b",
    r"\bblood\s+thinner\b",
    r"\banticoagulant\b",
    r"\bwarfarin\b",
    r"\bapixaban\b",
    r"\brivaroxaban\b",

    # Hinglish / Roman Hindi
    r"\bbaccha\b",
    r"\bbachcha\b",
    r"\bbacche\b",
    r"\bbachchon\b",
    r"\bpregnant\b",
    r"\bgarbhavati\b",
    r"\bstanpan\b",
    r"\bbreastfeed\b",
    r"\bkidney\s+ki\s+problem\b",
    r"\bliver\s+ki\s+problem\b",
    r"\bdawai\s+se\s+allergy\b",
    r"\bdava\s+se\s+allergy\b",
    r"\bkhoon\s+patla\s+karne\s+ki\s+dawai\b",

    # Hindi
    r"बच्चा",
    r"बच्चे",
    r"गर्भवती",
    r"गर्भावस्था",
    r"स्तनपान",
    r"किडनी",
    r"गुर्दे",
    r"लिवर",
    r"दवा से एलर्जी",
    r"खून पतला करने की दवा",

    # Telugu
    r"పిల్ల",
    r"గర్భవతి",
    r"గర్భం",
    r"స్తన్యపానం",
    r"మూత్రపిండ",
    r"కాలేయ",
    r"మందుకు అలర్జీ",

    # Kannada
    r"ಮಗು",
    r"ಗರ್ಭಿಣಿ",
    r"ಗರ್ಭಾವಸ್ಥೆ",
    r"ಸ್ತನ್ಯಪಾನ",
    r"ಮೂತ್ರಪಿಂಡ",
    r"ಯಕೃತ್ತು",
    r"ಔಷಧಿಗೆ ಅಲರ್ಜಿ",

    # Tamil
    r"குழந்தை",
    r"கர்ப்பிணி",
    r"கர்ப்பம்",
    r"தாய்ப்பால்",
    r"சிறுநீரகம்",
    r"கல்லீரல்",
    r"மருந்து ஒவ்வாமை",

    # Malayalam
    r"കുട്ടി",
    r"ഗർഭിണി",
    r"ഗർഭം",
    r"മുലയൂട്ടൽ",
    r"വൃക്ക",
    r"കരൾ",
    r"മരുന്ന് അലർജി",
]

SELF_HARM_PATTERNS_HINGLISH = [
    r"\bmarna\s+(hai|chahta|chahti)\b",
    r"\bmar\s+jaana\s+chahta\b",
    r"\bmar\s+jaana\s+chahti\b",
    r"\bjeena\s+nahi\s+chahta\b",
    r"\bjeena\s+nahi\s+chahti\b",
    r"\bjeene\s+ka\s+(man|dil)\s+nahi\b",
    r"\bjeene\s+ki\s+wajah\s+nahi\b",
    r"\bkhud\s+ko\s+khatam\b",
    r"\bkhud\s+ko\s+nuksan\b",
    r"\bkhud\s+ko\s+chot\b",
    r"\bapni\s+jaan\s+(dena|lena)\b",
    r"\bzindagi\s+khatam\s+kar\b",
    r"\bsuicide\s+karna\b",
    r"\baatmahatya\b",
    r"\bab\s+aur\s+nahi\s+jiya\s+jata\b",
    r"\bkoi\s+faayda\s+nahi\s+jeene\s+ka\b",
]


# --- Red-flag symptom patterns -> force URGENT regardless of anything else ---
# English patterns first, then per-language patterns below (checked together -
# see pre_check(). Intentionally not routed through a translation step: safety
# detection should not depend on translation quality/availability.
URGENT_PATTERNS = [
    r"\bchest pain\b", r"\bcan'?t breathe\b", r"\bbreathless(ness)?\b",
    r"\bsevere bleeding\b", r"\bunconscious(ness)?\b", r"\bfainted\b",
    r"\bstroke\b", r"\bparaly[sz]ed?\b", r"\bslurred speech\b",
    r"\bsevere abdominal pain\b", r"\bcoughing blood\b", r"\bblue lips\b",
    r"\bhigh fever\b.*\binfant\b", r"\bnewborn\b.*\bfever\b",
    r"\bseizure\b", r"\bfit(s)?\b.*\b(child|baby|infant)\b",
    r"\bsevere allergic reaction\b", r"\banaphylaxis\b",
    r"\bpregnan\w*\b.*\b(bleeding|severe pain)\b",
]

# --- Self-harm / crisis patterns -> bypass the LLM entirely ---
# NOTE: this list is intentionally broad and will over-trigger on some
# borderline phrasing - for this use case a false positive (extra crisis
# message) is far cheaper than a false negative, so bias towards recall.
SELF_HARM_PATTERNS = [
    r"\bsuicid\w*\b", r"\bkill myself\b", r"\bend my life\b", r"\bend it all\b",
    r"\bself.?harm\b", r"\bwant to die\b", r"\bno reason to live\b",
    r"\bdon'?t want to live\b", r"\bdo n'?t want to live\b",
    r"\bno point (in )?(living|life)\b", r"\bno point$", r"\bthere'?s no point\b",
    r"\bcan'?t (go on|take it anymore|take this anymore)\b",
    r"\bbetter off (dead|without me)\b", r"\bhurt(ing)? myself\b",
    r"\bnot worth living\b", r"\bgive up on life\b",
]

# --- Native-language red-flag patterns ---
# IMPORTANT: these were drafted by an AI assistant, NOT verified by a native
# speaker or clinical/crisis-language reviewer. Regional dialects, code-mixing
# (very common in real WhatsApp messages - e.g. Hindi+English in one sentence),
# and colloquial phrasing for these topics vary a lot. Treat this list as a
# first draft ONLY - it MUST be reviewed and expanded by native speakers
# (ideally with mental-health/clinical-language experience) before launch.
# Until then, keep the bias towards over-triggering (see note above) rather
# than trying to make these patterns "precise".
URGENT_PATTERNS_HI = [
    r"सीने में दर्द", r"सांस नहीं", r"साँस लेने में", r"बेहोश", r"लकवा",
    r"तेज़ बुखार", r"तेज बुखार", r"खून बह रहा", r"दौरा पड़", r"आवाज़ लड़खड़ा",
]
SELF_HARM_PATTERNS_HI = [
    r"आत्महत्या", r"मरना चाहता", r"मरना चाहती", r"जीने का मन नहीं",
    r"जीना नहीं चाहता", r"जीना नहीं चाहती", r"खुद को नुकसान", r"खुद को चोट",
]
URGENT_PATTERNS_TA = [
    r"மார்பு வலி", r"மூச்சு விட முடியவில்லை", r"மயங்கி", r"பக்கவாதம்",
    r"கடுமையான ரத்தப்போக்கு", r"வலிப்பு",
]
SELF_HARM_PATTERNS_TA = [
    r"தற்கொலை", r"சாக வேண்டும்", r"வாழ்க்கை வேண்டாம்", r"என்னை காயப்படுத்த",
]
URGENT_PATTERNS_TE = [
    r"ఛాతీ నొప్పి", r"ఊపిరి ఆడటం లేదు", r"స్పృహ కోల్పోయ", r"పక్షవాతం",
    r"తీవ్రమైన రక్తస్రావం", r"మూర్ఛ",
]
SELF_HARM_PATTERNS_TE = [
    r"ఆత్మహత్య", r"చనిపోవాలని", r"బ్రతకాలని లేదు", r"నన్ను నేను గాయపరచుకో",
]
URGENT_PATTERNS_KN = [
    r"ಎದೆ ನೋವು", r"ಉಸಿರಾಡಲು ಆಗುತ್ತಿಲ್ಲ", r"ಪ್ರಜ್ಞೆ ತಪ್ಪ", r"ಪಾರ್ಶ್ವವಾಯು",
    r"ತೀವ್ರ ರಕ್ತಸ್ರಾವ", r"ಫಿಟ್ಸ್",
]
SELF_HARM_PATTERNS_KN = [
    r"ಆತ್ಮಹತ್ಯೆ", r"ಸಾಯಬೇಕು ಅನ್ನಿಸು", r"ಬದುಕಬೇಕು ಅನ್ನಿಸುತ್ತಿಲ್ಲ", r"ನನ್ನನ್ನು ನಾನೇ ಗಾಯಗೊಳಿಸ",
]
URGENT_PATTERNS_ML = [
    r"നെഞ്ചുവേദന", r"ശ്വാസം കിട്ടുന്നില്ല", r"ബോധം കെട്ടു", r"പക്ഷാഘാതം",
    r"കടുത്ത രക്തസ്രാവം", r"അപസ്മാരം",
]
SELF_HARM_PATTERNS_ML = [
    r"ആത്മഹത്യ", r"മരിക്കണം എന്ന്", r"ജീവിക്കാൻ തോന്നുന്നില്ല", r"സ്വയം മുറിവേൽപ്പിക്ക",
]

_URGENT_BY_LANG = {
    "hi": URGENT_PATTERNS_HI, "hinglish": URGENT_PATTERNS_HINGLISH, "ta": URGENT_PATTERNS_TA,
    "te": URGENT_PATTERNS_TE, "kn": URGENT_PATTERNS_KN, "ml": URGENT_PATTERNS_ML,}

_SELF_HARM_BY_LANG = {
    "hi": SELF_HARM_PATTERNS_HI, "hinglish": SELF_HARM_PATTERNS_HINGLISH, "ta": SELF_HARM_PATTERNS_TA,
    "te": SELF_HARM_PATTERNS_TE, "kn": SELF_HARM_PATTERNS_KN, "ml": SELF_HARM_PATTERNS_ML,
}

# --- Moderate-risk indicators -> recommend doctor consult ---
MODERATE_PATTERNS = [
    r"\bdiabetes\b", r"\bblood pressure\b", r"\bhypertension\b",
    r"\brecurring\b", r"\bchronic\b", r"\bfor (a )?(week|weeks|days)\b",
    r"\bworsening\b", r"\bnot improving\b",
]

# --- Output validator: banned patterns the LLM must never produce ---
BANNED_OUTPUT_PATTERNS = [
    r"\byou have\b(?:\s+\w+){0,4}\s+(disease|infection|cancer|disorder|syndrome|tumou?r)",  # diagnostic language
    r"\byou (are|'re) suffering from\b",
    r"\bthis confirms\b", r"\byou definitely have\b",
    r"\btake \d+\s?(mg|ml|mcg|tablet|tablets|pills?|drops?|capsules?)\b",  # dosage instructions
    r"\bstop taking\b", r"\bincrease the dose\b", r"\bdouble the dose\b",
    r"\breduce the dose\b", r"\bchange (your|the) medication\b",
    r"\bI (diagnose|confirm) you\b",
    # Block personalised medicine directives while still allowing neutral
    # product discovery such as "you can explore OTC options".
    r"\byou should (take|use|start)\b",
    r"\byou need to (take|use|start)\b",
    r"\bi recommend (that you )?(take|use|start)\b",
    r"\bstart taking this (medicine|medication|tablet|supplement)\b",
]

CRISIS_MESSAGE = {
    "en": (
        "I'm really glad you reached out. Please talk to someone right now — "
        "you can call the KIRAN mental health helpline (toll-free): 1800-599-0019, "
        "available 24/7 in multiple languages. If you are in immediate danger, please "
        "call 112 or go to the nearest hospital. You don't have to go through this alone."
    ),
    # NOTE: these translations were drafted by an AI assistant, not a native-speaker
    # medical/crisis-communications reviewer. Given how much precision and tone matter
    # for crisis messaging, get each one checked by a native speaker (ideally someone
    # with mental-health messaging experience) before this goes anywhere near real users.
    "hi": (
        "मुझे खुशी है कि आपने बात की। कृपया अभी किसी से बात करें — आप KIRAN मानसिक स्वास्थ्य "
        "हेल्पलाइन (टोल-फ्री) पर कॉल कर सकते हैं: 1800-599-0019, जो 24/7 कई भाषाओं में उपलब्ध है। "
        "अगर आप तुरंत खतरे में हैं, तो कृपया 112 पर कॉल करें या नज़दीकी अस्पताल जाएं। "
        "आपको यह अकेले नहीं झेलना है।"
    ),
    "ta": (
        "நீங்கள் தொடர்பு கொண்டதில் மகிழ்ச்சி. தயவுசெய்து இப்போது யாரிடமாவது பேசுங்கள் — "
        "KIRAN மனநல உதவி எண்ணை (இலவசம்) அழைக்கலாம்: 1800-599-0019, பல மொழிகளில் 24/7 கிடைக்கும். "
        "உடனடி ஆபத்தில் இருந்தால், 112ஐ அழைக்கவும் அல்லது அருகிலுள்ள மருத்துவமனைக்குச் செல்லவும். "
        "இதை தனியாக சமாளிக்க வேண்டாம்."
    ),
    "te": (
        "మీరు మాట్లాడినందుకు సంతోషంగా ఉంది. దయచేసి ఇప్పుడే ఎవరితోనైనా మాట్లాడండి — "
        "KIRAN మానసిక ఆరోగ్య హెల్ప్‌లైన్‌కు (టోల్-ఫ్రీ) కాల్ చేయవచ్చు: 1800-599-0019, "
        "24/7 అనేక భాషల్లో అందుబాటులో ఉంది. తక్షణ ప్రమాదంలో ఉంటే, దయచేసి 112కు కాల్ చేయండి "
        "లేదా సమీప ఆసుపత్రికి వెళ్లండి. దీన్ని మీరు ఒంటరిగా ఎదుర్కోవాల్సిన అవసరం లేదు."
    ),
    "kn": (
        "ನೀವು ಸಂಪರ್ಕಿಸಿದ್ದಕ್ಕೆ ಸಂತೋಷ. ದಯವಿಟ್ಟು ಈಗಲೇ ಯಾರೊಂದಿಗಾದರೂ ಮಾತನಾಡಿ — "
        "KIRAN ಮಾನಸಿಕ ಆರೋಗ್ಯ ಸಹಾಯವಾಣಿಗೆ (ಉಚಿತ) ಕರೆ ಮಾಡಬಹುದು: 1800-599-0019, "
        "24/7 ಹಲವು ಭಾಷೆಗಳಲ್ಲಿ ಲಭ್ಯವಿದೆ. ತಕ್ಷಣದ ಅಪಾಯದಲ್ಲಿದ್ದರೆ, ದಯವಿಟ್ಟು 112 ಗೆ ಕರೆ ಮಾಡಿ "
        "ಅಥವಾ ಹತ್ತಿರದ ಆಸ್ಪತ್ರೆಗೆ ಹೋಗಿ. ಇದನ್ನು ನೀವು ಒಬ್ಬಂಟಿಯಾಗಿ ಎದುರಿಸಬೇಕಿಲ್ಲ."
    ),
    "ml": (
        "നിങ്ങൾ സംസാരിച്ചതിൽ സന്തോഷം. ദയവായി ഇപ്പോൾ തന്നെ ആരോടെങ്കിലും സംസാരിക്കുക — "
        "KIRAN മാനസികാരോഗ്യ ഹെൽപ്പ്‌ലൈനിലേക്ക് (ടോൾ-ഫ്രീ) വിളിക്കാം: 1800-599-0019, "
        "24/7 പല ഭാഷകളിലും ലഭ്യമാണ്. ഉടനടി അപകടത്തിലാണെങ്കിൽ, ദയവായി 112ലേക്ക് വിളിക്കുക "
        "അല്ലെങ്കിൽ അടുത്തുള്ള ആശുപത്രിയിൽ പോകുക. ഇത് നിങ്ങൾ ഒറ്റയ്ക്ക് നേരിടേണ്ടതില്ല."
    ),
}

EMERGENCY_MESSAGE = {
    "en": (
        "⚠️ Based on what you've described, this could be serious and needs immediate "
        "medical attention. Please call 112 (national emergency) or 108 (ambulance) right "
        "away, or go to the nearest hospital / emergency room. Please do not wait.\n\n"
        "I'm sending you tap-to-call contacts below, and I can also help you find the "
        "nearest hospital if you share your location."
    ),
    "hinglish": (
        "⚠️ Aapne jo bataya hai, uske hisaab se yeh serious ho sakta hai aur turant medical "
        "help ki zarurat ho sakti hai. Abhi 112 (national emergency) ya 108 (ambulance) par "
        "call karein, ya nearest hospital/emergency room jaayein. Please wait na karein.\n\n"
        "Main neeche emergency call options bhej raha hoon. Is situation mein OTC product ya "
        "paid consultation se pehle emergency care ko priority dein."
    ),
    # Same caveat as above: AI-drafted, needs native-speaker medical review before launch.
    "hi": (
        "⚠️ आपने जो बताया है, उसके आधार पर यह गंभीर हो सकता है और तुरंत चिकित्सा सहायता की "
        "आवश्यकता है। कृपया अभी 112 (राष्ट्रीय आपातकाल) या 108 (एम्बुलेंस) पर कॉल करें, या "
        "नज़दीकी अस्पताल/इमरजेंसी रूम जाएं। कृपया इंतज़ार न करें।\n\n"
        "मैं नीचे तुरंत कॉल करने के लिए संपर्क भेज रहा/रही हूं, और अगर आप अपना लोकेशन शेयर करें "
        "तो मैं नज़दीकी अस्पताल ढूंढने में भी मदद कर सकता/सकती हूं।"
    ),
    "ta": (
        "⚠️ நீங்கள் விவரித்தவற்றின் அடிப்படையில், இது கடுமையானதாக இருக்கலாம், உடனடி மருத்துவ "
        "கவனிப்பு தேவை. தயவுசெய்து இப்போதே 112 (தேசிய அவசரகாலம்) அல்லது 108 (ஆம்புலன்ஸ்) ஐ "
        "அழைக்கவும், அல்லது அருகிலுள்ள மருத்துவமனை/அவசர சிகிச்சைப் பிரிவிற்குச் செல்லவும். "
        "தயவுசெய்து காத்திருக்க வேண்டாம்.\n\nகீழே உடனடியாக அழைக்கும் தொடர்புகளை அனுப்புகிறேன், "
        "நீங்கள் உங்கள் இருப்பிடத்தைப் பகிர்ந்தால் அருகிலுள்ள மருத்துவமனையையும் கண்டுபிடிக்க உதவ முடியும்."
    ),
    "te": (
        "⚠️ మీరు వివరించిన దాని ఆధారంగా, ఇది తీవ్రమైనది కావచ్చు మరియు తక్షణ వైద్య సహాయం అవసరం. "
        "దయచేసి వెంటనే 112 (జాతీయ అత్యవసరం) లేదా 108 (అంబులెన్స్)కు కాల్ చేయండి, లేదా సమీప "
        "ఆసుపత్రి/అత్యవసర గదికి వెళ్లండి. దయచేసి వేచి ఉండవద్దు.\n\nకింద తక్షణమే కాల్ చేయగల "
        "కాంటాక్ట్‌లను పంపుతున్నాను, మీరు మీ లొకేషన్ షేర్ చేస్తే సమీప ఆసుపత్రిని కనుగొనడంలో కూడా సహాయం చేయగలను."
    ),
    "kn": (
        "⚠️ ನೀವು ವಿವರಿಸಿದ್ದರ ಆಧಾರದ ಮೇಲೆ, ಇದು ಗಂಭೀರವಾಗಿರಬಹುದು ಮತ್ತು ತಕ್ಷಣದ ವೈದ್ಯಕೀಯ ಸಹಾಯ "
        "ಅಗತ್ಯವಿದೆ. ದಯವಿಟ್ಟು ಈಗಲೇ 112 (ರಾಷ್ಟ್ರೀಯ ತುರ್ತು) ಅಥವಾ 108 (ಆಂಬ್ಯುಲೆನ್ಸ್) ಗೆ ಕರೆ ಮಾಡಿ, "
        "ಅಥವಾ ಹತ್ತಿರದ ಆಸ್ಪತ್ರೆ/ತುರ್ತು ವಿಭಾಗಕ್ಕೆ ಹೋಗಿ. ದಯವಿಟ್ಟು ಕಾಯಬೇಡಿ.\n\nಕೆಳಗೆ ತಕ್ಷಣ ಕರೆ "
        "ಮಾಡಬಹುದಾದ ಸಂಪರ್ಕಗಳನ್ನು ಕಳುಹಿಸುತ್ತಿದ್ದೇನೆ, ನೀವು ನಿಮ್ಮ ಸ್ಥಳವನ್ನು ಹಂಚಿಕೊಂಡರೆ ಹತ್ತಿರದ "
        "ಆಸ್ಪತ್ರೆಯನ್ನು ಹುಡುಕಲು ಸಹ ಸಹಾಯ ಮಾಡಬಲ್ಲೆ."
    ),
    "ml": (
        "⚠️ നിങ്ങൾ വിവരിച്ചതിന്റെ അടിസ്ഥാനത്തിൽ, ഇത് ഗുരുതരമാകാം, ഉടനടി വൈദ്യസഹായം ആവശ്യമാണ്. "
        "ദയവായി ഉടൻ 112 (ദേശീയ അടിയന്തരം) അല്ലെങ്കിൽ 108 (ആംബുലൻസ്) ലേക്ക് വിളിക്കുക, അല്ലെങ്കിൽ "
        "അടുത്തുള്ള ആശുപത്രി/എമർജൻസി റൂമിൽ പോകുക. ദയവായി കാത്തിരിക്കരുത്.\n\nതാഴെ ഉടനടി "
        "വിളിക്കാവുന്ന കോൺടാക്റ്റുകൾ അയക്കുന്നു, നിങ്ങളുടെ ലൊക്കേഷൻ പങ്കിട്ടാൽ അടുത്തുള്ള ആശുപത്രി "
        "കണ്ടെത്താനും സഹായിക്കാം."
    ),
}

DISCLAIMER = {
    "en": "\n\n_Health information only — not a diagnosis or prescription._",
    "hi": "\n\n_यह सामान्य स्वास्थ्य जानकारी है — निदान या प्रिस्क्रिप्शन नहीं।_",
    "ta": "\n\n_இது பொதுவான சுகாதார தகவல் — நோயறிதல் அல்லது மருந்துச் சீட்டு அல்ல._",
    "te": "\n\n_ఇది సాధారణ ఆరోగ్య సమాచారం — నిర్ధారణ లేదా ప్రిస్క్రిప్షన్ కాదు._",
    "kn": "\n\n_ಇದು ಸಾಮಾನ್ಯ ಆರೋಗ್ಯ ಮಾಹಿತಿ — ರೋಗನಿರ್ಣಯ ಅಥವಾ ಪ್ರಿಸ್ಕ್ರಿಪ್ಷನ್ ಅಲ್ಲ._",
    "ml": "\n\n_ഇത് പൊതുവായ ആരോഗ്യ വിവരമാണ് — രോഗനിർണയമോ കുറിപ്പടിയോ അല്ല._",
}


def _matches_any(text: str, patterns: list[str]) -> str | None:
    text_l = text.lower()
    for p in patterns:
        if re.search(p, text_l):
            return p
    return None

def has_high_risk_medicine_context(text: str) -> bool:
    return _matches_any(text, HIGH_RISK_MEDICINE_PATTERNS) is not None


def pre_check(user_text: str) -> TriageResult:
    """Run before any LLM call. Can short-circuit the whole pipeline.
    Checks English patterns AND all native-language pattern lists together
    (not just the detected language) - WhatsApp messages are frequently
    code-mixed (e.g. Hindi+English in one sentence), so restricting to a
    single detected language risks missing a red flag written in the other."""
    all_urgent = URGENT_PATTERNS + [p for plist in _URGENT_BY_LANG.values() for p in plist]
    all_self_harm = SELF_HARM_PATTERNS + [p for plist in _SELF_HARM_BY_LANG.values() for p in plist]

    if (m := _matches_any(user_text, all_self_harm)):
        return TriageResult(Tier.SELF_HARM, m)
    if (m := _matches_any(user_text, all_urgent)):
        return TriageResult(Tier.URGENT, m)
    if (m := _matches_any(user_text, MODERATE_PATTERNS)):
        return TriageResult(Tier.MODERATE, m)
    return TriageResult(Tier.LOW, None)


def validate_output(generated_text: str) -> tuple[bool, str | None]:
    """Run on every LLM response before sending to user.
    Returns (is_safe, matched_banned_pattern)."""
    m = _matches_any(generated_text, BANNED_OUTPUT_PATTERNS)
    return (m is None, m)


def apply_disclaimer(text: str, lang: str = "en") -> str:
    return text + DISCLAIMER.get(lang, DISCLAIMER["en"])
