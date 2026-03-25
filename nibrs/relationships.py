IPV_RELATIONS = [
    'victim was spouse',
    'victim was common-law spouse',
    'victim was ex-spouse',
    'victim was boyfriend/girlfriend',
]

# ── All DV relationships (for broader DV flags) ──
DV_RELATIONS_ALL = IPV_RELATIONS + [
    'victim was child',
    'victim was step-child',
    'victim was child of boyfriend/girlfriend',
    'victim was parent',
    'victim was sibling (brother or sister)',
    'victim was step-parent',
    'victim was step-sibling (stepbrother or stepsister)',
    'victim was grandparent',
    'victim was grandchild',
    'victim was in-law',
    'victim was other family member',
    # NOTE: 'victim was other - known to victim' deliberately EXCLUDED
    # It is NOT a family relationship — FIX #2
]

# ── FIXED otherfam values (FIX #2) ──
OTHERFAM_VALUES = [
    'victim was parent',
    'victim was sibling (brother or sister)',
    'victim was grandparent',
    'victim was in-law',
    'victim was step-parent',
    'victim was step-sibling (stepbrother or stepsister)',
    'victim was other family member',
    # 'victim was other - known to victim' REMOVED — not family
]

CHILD_VALUES = [
    'victim was child',
    'victim was grandchild',
    'victim was step-child',
    'victim was child of boyfriend/girlfriend',
]

KNOWN_VALUES = [
    'victim was acquaintance',
    'victim was friend',
    'victim was neighbor',
    'victim was babysittee (the baby)',
    'victim was employee',
    'victim was employer',
    'victim was otherwise known',
    'victim was other - known to victim',  
]

SERIOUS_INJURIES = [
    'apparent broken bones', 'other major injury', 'possible internal injury',
    'severe laceration', 'loss of teeth', 'unconsciousness',
]