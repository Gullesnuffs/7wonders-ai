import copy
from card import Card, Color, Constant, Cost, ScoreEffect, ProductionEffect, Resource, Wonder, GoldEffect, CardCounter, TradingEffect, ScienceEffect, Science, MilitaryEffect, DefeatCounter, WonderStage, WonderCounter


def getCards(age, players):
    if (age == 1):
        cards = AGE1_3PLAYERS
    if (age == 2):
        cards = AGE2_3PLAYERS
    if (age == 3):
        cards = AGE3_3PLAYERS
    return copy.copy(cards)


MANUFACTURED_RESOURCES = [Resource.GLASS, Resource.CLOTH, Resource.PAPYRUS]
NONMANUFACTURED_RESOURCES = [Resource.CLAY, Resource.STONE, Resource.WOOD, Resource.ORE]

DEFAULT = Card(name="Default", color=Color.BROWN)

# Green cards
# Age 1
APOTHECARY = Card(
    name="Apothecary",
    color=Color.GREEN,
    cost=Cost(resources=[Resource.CLOTH]),
    effects=[
        ScienceEffect([Science.COMPASS]),
    ])

WORKSHOP = Card(
    name="Workshop",
    color=Color.GREEN,
    cost=Cost(resources=[Resource.GLASS]),
    effects=[
        ScienceEffect([Science.COG]),
    ])

SCRIPTORIUM = Card(
    name="Scriptorium",
    color=Color.GREEN,
    cost=Cost(resources=[Resource.PAPYRUS]),
    effects=[
        ScienceEffect([Science.TABLET]),
    ])

# Age 2
DISPENSARY = Card(
    name="Dispensary",
    color=Color.GREEN,
    cost=Cost(resources=[Resource.ORE, Resource.ORE, Resource.GLASS]),
    chainFrom=[APOTHECARY],
    effects=[
        ScienceEffect([Science.COMPASS]),
    ])

LABORATORY = Card(
    name="Laboratory",
    color=Color.GREEN,
    cost=Cost(resources=[Resource.CLAY, Resource.CLAY, Resource.PAPYRUS]),
    chainFrom=[WORKSHOP],
    effects=[
        ScienceEffect([Science.COG]),
    ])

LIBRARY = Card(
    name="Library",
    color=Color.GREEN,
    cost=Cost(resources=[Resource.STONE, Resource.STONE, Resource.CLOTH]),
    chainFrom=[SCRIPTORIUM],
    effects=[
        ScienceEffect([Science.TABLET]),
    ])

SCHOOL = Card(
    name="School",
    color=Color.GREEN,
    cost=Cost(resources=[Resource.WOOD, Resource.PAPYRUS]),
    effects=[
        ScienceEffect([Science.TABLET]),
    ])

# Age 3
LODGE = Card(
    name="Lodge",
    color=Color.GREEN,
    cost=Cost(resources=[Resource.CLAY, Resource.CLAY, Resource.CLOTH, Resource.PAPYRUS]),
    chainFrom=[DISPENSARY],
    effects=[
        ScienceEffect([Science.COMPASS]),
    ])

OBSERVATORY = Card(
    name="Observatory",
    color=Color.GREEN,
    cost=Cost(resources=[Resource.ORE, Resource.ORE, Resource.GLASS, Resource.CLOTH]),
    chainFrom=[LABORATORY],
    effects=[
        ScienceEffect([Science.COG]),
    ])

UNIVERSITY = Card(
    name="University",
    color=Color.GREEN,
    cost=Cost(resources=[Resource.WOOD, Resource.WOOD, Resource.PAPYRUS, Resource.GLASS]),
    chainFrom=[LIBRARY],
    effects=[
        ScienceEffect([Science.TABLET]),
    ])

ACADEMY = Card(
    name="Academy",
    color=Color.GREEN,
    cost=Cost(resources=[Resource.STONE, Resource.STONE, Resource.STONE, Resource.GLASS]),
    chainFrom=[SCHOOL],
    effects=[
        ScienceEffect([Science.COMPASS]),
    ])

STUDY = Card(
    name="Study",
    color=Color.GREEN,
    cost=Cost(resources=[Resource.WOOD, Resource.PAPYRUS, Resource.CLOTH]),
    chainFrom=[SCHOOL],
    effects=[
        ScienceEffect([Science.COG]),
    ])

# Red cards
# Age 1
STOCKADE = Card(
    name="Stockade",
    color=Color.RED,
    cost=Cost(resources=[Resource.WOOD]),
    effects=[
        MilitaryEffect(1),
    ])

BARRACKS = Card(
    name="Barracks",
    color=Color.RED,
    cost=Cost(resources=[Resource.ORE]),
    effects=[
        MilitaryEffect(1),
    ])

GUARD_TOWER = Card(
    name="Guard tower",
    color=Color.RED,
    cost=Cost(resources=[Resource.CLAY]),
    effects=[
        MilitaryEffect(1),
    ])

# Age 2
WALLS = Card(
    name="Walls",
    color=Color.RED,
    cost=Cost(resources=[Resource.STONE, Resource.STONE, Resource.STONE]),
    effects=[
        MilitaryEffect(2),
    ])

TRAINING_GROUND = Card(
    name="Training ground",
    color=Color.RED,
    cost=Cost(resources=[Resource.WOOD, Resource.ORE, Resource.ORE]),
    effects=[
        MilitaryEffect(2),
    ])

STABLES = Card(
    name="Stables",
    color=Color.RED,
    cost=Cost(resources=[Resource.ORE, Resource.CLAY, Resource.WOOD]),
    chainFrom=[APOTHECARY],
    effects=[
        MilitaryEffect(2),
    ])

ARCHERY_RANGE = Card(
    name="Archery range",
    color=Color.RED,
    cost=Cost(resources=[Resource.WOOD, Resource.WOOD, Resource.ORE]),
    chainFrom=[WORKSHOP],
    effects=[
        MilitaryEffect(2),
    ])

# Age 3
FORTIFICATIONS = Card(
    name="Fortifications",
    color=Color.RED,
    cost=Cost(resources=[Resource.STONE, Resource.ORE, Resource.ORE, Resource.ORE]),
    chainFrom=[WALLS],
    effects=[
        MilitaryEffect(3),
    ])

CIRCUS = Card(
    name="Circus",
    color=Color.RED,
    cost=Cost(resources=[Resource.STONE, Resource.STONE, Resource.STONE, Resource.ORE]),
    chainFrom=[TRAINING_GROUND],
    effects=[
        MilitaryEffect(3),
    ])

ARSENAL = Card(
    name="Arsenal",
    color=Color.RED,
    cost=Cost(resources=[Resource.ORE, Resource.WOOD, Resource.WOOD, Resource.CLOTH]),
    effects=[
        MilitaryEffect(3),
    ])

SIEGE_WORKSHOP = Card(
    name="Siege workshop",
    color=Color.RED,
    cost=Cost(resources=[Resource.WOOD, Resource.CLAY, Resource.CLAY, Resource.CLAY]),
    chainFrom=[LABORATORY],
    effects=[
        MilitaryEffect(3),
    ])

# Blue cards
# Age 1
ALTAR = Card(
    name="Altar",
    color=Color.BLUE,
    effects=[
        ScoreEffect(Constant(value=2)),
    ])

BATHS = Card(
    name="Baths",
    color=Color.BLUE,
    cost=Cost(resources=[Resource.STONE]),
    effects=[
        ScoreEffect(Constant(value=3)),
    ])

PAWNSHOP = Card(
    name="Pawnshop",
    color=Color.BLUE,
    effects=[
        ScoreEffect(Constant(value=3)),
    ])

THEATER = Card(
    name="Theater",
    color=Color.BLUE,
    effects=[
        ScoreEffect(Constant(value=2)),
    ])

# Age 2
AQUEDUCT = Card(
    name="Aqueduct",
    color=Color.BLUE,
    cost=Cost(resources=[Resource.STONE, Resource.STONE, Resource.STONE]),
    chainFrom=[BATHS],
    effects=[
        ScoreEffect(Constant(value=5)),
    ])

TEMPLE = Card(
    name="Temple",
    color=Color.BLUE,
    cost=Cost(resources=[Resource.WOOD, Resource.CLAY, Resource.GLASS]),
    chainFrom=[ALTAR],
    effects=[
        ScoreEffect(Constant(value=3)),
    ])

STATUE = Card(
    name="Statue",
    color=Color.BLUE,
    cost=Cost(resources=[Resource.WOOD, Resource.ORE, Resource.ORE]),
    chainFrom=[THEATER],
    effects=[
        ScoreEffect(Constant(value=4)),
    ])

COURTHOUSE = Card(
    name="Courthouse",
    color=Color.BLUE,
    cost=Cost(resources=[Resource.CLAY, Resource.CLAY, Resource.CLOTH]),
    chainFrom=[SCRIPTORIUM],
    effects=[
        ScoreEffect(Constant(value=4)),
    ])

# Age 3
PANTHEON = Card(
    name="Pantheon",
    color=Color.BLUE,
    cost=Cost(resources=[Resource.CLAY, Resource.CLAY, Resource.ORE, Resource.PAPYRUS, Resource.CLOTH, Resource.GLASS]),
    chainFrom=[TEMPLE],
    effects=[
        ScoreEffect(Constant(value=7)),
    ])

GARDENS = Card(
    name="Gardens",
    color=Color.BLUE,
    cost=Cost(resources=[Resource.WOOD, Resource.CLAY, Resource.CLAY]),
    chainFrom=[STATUE],
    effects=[
        ScoreEffect(Constant(value=5)),
    ])

TOWN_HALL = Card(
    name="Town hall",
    color=Color.BLUE,
    cost=Cost(resources=[Resource.GLASS, Resource.ORE, Resource.STONE, Resource.STONE]),
    effects=[
        ScoreEffect(Constant(value=6)),
    ])

PALACE = Card(
    name="Palace",
    color=Color.BLUE,
    cost=Cost(resources=[Resource.GLASS, Resource.PAPYRUS, Resource.CLOTH, Resource.CLAY, Resource.WOOD, Resource.ORE, Resource.STONE]),
    effects=[
        ScoreEffect(Constant(value=8)),
    ])

SENATE = Card(
    name="Senate",
    color=Color.BLUE,
    cost=Cost(resources=[Resource.ORE, Resource.STONE, Resource.WOOD, Resource.WOOD]),
    chainFrom=[LIBRARY],
    effects=[
        ScoreEffect(Constant(value=6)),
    ])

# Brown cards
# Age 1
LUMBER_YARD = Card(
    name="Lumber yard",
    color=Color.BROWN,
    effects=[
        ProductionEffect([Resource.WOOD]),
    ])

STONE_PIT = Card(
    name="Stone pit",
    color=Color.BROWN,
    effects=[
        ProductionEffect([Resource.STONE]),
    ])

CLAY_POOL = Card(
    name="Clay pool",
    color=Color.BROWN,
    effects=[
        ProductionEffect([Resource.CLAY]),
    ])

ORE_VEIN = Card(
    name="Ore vein",
    color=Color.BROWN,
    effects=[
        ProductionEffect([Resource.ORE]),
    ])

CLAY_PIT = Card(
    name="Clay pit",
    color=Color.BROWN,
    cost=Cost(gold=1),
    effects=[
        ProductionEffect([Resource.CLAY, Resource.ORE]),
    ])

TIMBER_YARD = Card(
    name="Timber yard",
    color=Color.BROWN,
    cost=Cost(gold=1),
    effects=[
        ProductionEffect([Resource.STONE, Resource.WOOD]),
    ])

TREE_FARM = Card(
    name="Tree farm",
    color=Color.BROWN,
    cost=Cost(gold=1),
    effects=[
        ProductionEffect([Resource.WOOD, Resource.CLAY]),
    ])

EXCAVATION = Card(
    name="Excavation",
    color=Color.BROWN,
    cost=Cost(gold=1),
    effects=[
        ProductionEffect([Resource.STONE, Resource.CLAY]),
    ])

FOREST_CAVE = Card(
    name="Forest cave",
    color=Color.BROWN,
    cost=Cost(gold=1),
    effects=[
        ProductionEffect([Resource.WOOD, Resource.ORE]),
    ])

MINE = Card(
    name="Mine",
    color=Color.BROWN,
    cost=Cost(gold=1),
    effects=[
        ProductionEffect([Resource.ORE, Resource.STONE]),
    ])

# Age 2
SAWMILL = Card(
    name="Sawmill",
    color=Color.BROWN,
    cost=Cost(gold=1),
    effects=[
        ProductionEffect([Resource.WOOD]),
        ProductionEffect([Resource.WOOD]),
    ])

QUARRY = Card(
    name="Quarry",
    color=Color.BROWN,
    cost=Cost(gold=1),
    effects=[
        ProductionEffect([Resource.STONE]),
        ProductionEffect([Resource.STONE]),
    ])

BRICKYARD = Card(
    name="Brickyard",
    color=Color.BROWN,
    cost=Cost(gold=1),
    effects=[
        ProductionEffect([Resource.CLAY]),
        ProductionEffect([Resource.CLAY]),
    ])

FOUNDRY = Card(
    name="Foundry",
    color=Color.BROWN,
    cost=Cost(gold=1),
    effects=[
        ProductionEffect([Resource.ORE]),
        ProductionEffect([Resource.ORE]),
    ])

# Grey cards
LOOM = Card(
    name="Loom",
    color=Color.GREY,
    effects=[
        ProductionEffect([Resource.CLOTH]),
    ])

GLASSWORKS = Card(
    name="Glassworks",
    color=Color.GREY,
    effects=[
        ProductionEffect([Resource.GLASS]),
    ])

PRESS = Card(
    name="Press",
    color=Color.GREY,
    effects=[
        ProductionEffect([Resource.PAPYRUS]),
    ])

# Yellow cards
# Age 1
TAVERN = Card(
    name="Tavern",
    color=Color.YELLOW,
    effects=[
        GoldEffect(Constant(value=5)),
    ])

EAST_TRADING_POST = Card(
    name="East trading post",
    color=Color.YELLOW,
    effects=[
        TradingEffect(resources=NONMANUFACTURED_RESOURCES, rightNeighbor=True),
    ])

WEST_TRADING_POST = Card(
    name="West trading post",
    color=Color.YELLOW,
    effects=[
        TradingEffect(resources=NONMANUFACTURED_RESOURCES, leftNeighbor=True),
    ])

MARKETPLACE = Card(
    name="Marketplace",
    color=Color.YELLOW,
    effects=[
        TradingEffect(resources=MANUFACTURED_RESOURCES, leftNeighbor=True, rightNeighbor=True),
    ])

# Age 2
FORUM = Card(
    name="Forum",
    color=Color.YELLOW,
    cost=Cost(resources=[Resource.CLAY, Resource.CLAY]),
    chainFrom=[EAST_TRADING_POST, WEST_TRADING_POST],
    effects=[
        ProductionEffect(MANUFACTURED_RESOURCES),
    ])

CARAVANSERY = Card(
    name="Caravansery",
    color=Color.YELLOW,
    cost=Cost(resources=[Resource.WOOD, Resource.WOOD]),
    chainFrom=[MARKETPLACE],
    effects=[
        ProductionEffect(NONMANUFACTURED_RESOURCES),
    ])

VINEYARD = Card(
    name="Vineyard",
    color=Color.YELLOW,
    effects=[
        GoldEffect(CardCounter(countSelf=True, countNeighbors=True, color=Color.BROWN)),
    ])

BAZAR = Card(
    name="Bazar",
    color=Color.YELLOW,
    effects=[
        GoldEffect(CardCounter(countSelf=True, countNeighbors=True, color=Color.GREY, multiplier=2)),
    ])


# Age 3
HAVEN = Card(
    name="Haven",
    color=Color.YELLOW,
    cost=Cost(resources=[Resource.CLOTH, Resource.ORE, Resource.WOOD]),
    chainFrom=[FORUM],
    effects=[
        GoldEffect(CardCounter(countSelf=True, color=Color.BROWN)),
        ScoreEffect(CardCounter(countSelf=True, color=Color.BROWN)),
    ])

LIGHTHOUSE = Card(
    name="Lighthouse",
    color=Color.YELLOW,
    cost=Cost(resources=[Resource.GLASS, Resource.STONE]),
    chainFrom=[CARAVANSERY],
    effects=[
        GoldEffect(CardCounter(countSelf=True, color=Color.YELLOW)),
        ScoreEffect(CardCounter(countSelf=True, color=Color.YELLOW)),
    ])

CHAMBER_OF_COMMERCE = Card(
    name="Chamber of commerce",
    color=Color.YELLOW,
    cost=Cost(resources=[Resource.CLAY, Resource.CLAY, Resource.PAPYRUS]),
    effects=[
        GoldEffect(CardCounter(countSelf=True, color=Color.GREY, multiplier=2)),
        ScoreEffect(CardCounter(countSelf=True, color=Color.GREY, multiplier=2)),
    ])

ARENA = Card(
    name="Arena",
    color=Color.YELLOW,
    cost=Cost(resources=[Resource.ORE, Resource.STONE, Resource.STONE]),
    chainFrom=[DISPENSARY],
    effects=[
        GoldEffect(WonderCounter(countSelf=True, multiplier=3)),
        ScoreEffect(WonderCounter(countSelf=True)),
    ])

# Purple cards
WORKERS_GUILD = Card(
    name="Workers' guild",
    color=Color.PURPLE,
    cost=Cost(resources=[Resource.ORE, Resource.ORE, Resource.CLAY, Resource.STONE, Resource.WOOD]),
    effects=[
        ScoreEffect(CardCounter(countNeighbors=True, color=Color.BROWN)),
    ])

CRAFTSMENS_GUILD = Card(
    name="Craftsmens' guild",
    color=Color.PURPLE,
    cost=Cost(resources=[Resource.ORE, Resource.ORE, Resource.STONE, Resource.STONE]),
    effects=[
        ScoreEffect(CardCounter(countNeighbors=True, color=Color.GREY, multiplier=2)),
    ])

TRADERS_GUILD = Card(
    name="Traders' guild",
    color=Color.PURPLE,
    cost=Cost(resources=[Resource.CLOTH, Resource.PAPYRUS, Resource.GLASS]),
    effects=[
        ScoreEffect(CardCounter(countNeighbors=True, color=Color.YELLOW)),
    ])

PHILOSOPHERS_GUILD = Card(
    name="Philosophers' guild",
    color=Color.PURPLE,
    cost=Cost(resources=[Resource.CLAY, Resource.CLAY, Resource.CLAY, Resource.CLOTH, Resource.PAPYRUS]),
    effects=[
        ScoreEffect(CardCounter(countNeighbors=True, color=Color.GREEN)),
    ])

SPIES_GUILD = Card(
    name="Spies' guild",
    color=Color.PURPLE,
    cost=Cost(resources=[Resource.CLAY, Resource.CLAY, Resource.CLAY, Resource.GLASS]),
    effects=[
        ScoreEffect(CardCounter(countNeighbors=True, color=Color.RED)),
    ])

STRATEGISTS_GUILD = Card(
    name="Strategists' guild",
    color=Color.PURPLE,
    cost=Cost(resources=[Resource.ORE, Resource.ORE, Resource.STONE, Resource.CLOTH]),
    effects=[
        ScoreEffect(DefeatCounter(countNeighbors=True)),
    ])

SHIPOWNERS_GUILD = Card(
    name="Shipowners' guild",
    color=Color.PURPLE,
    cost=Cost(resources=[Resource.WOOD, Resource.WOOD, Resource.WOOD, Resource.PAPYRUS, Resource.GLASS]),
    effects=[
        ScoreEffect(CardCounter(countSelf=True, color=Color.BROWN)),
        ScoreEffect(CardCounter(countSelf=True, color=Color.GREY)),
        ScoreEffect(CardCounter(countSelf=True, color=Color.PURPLE)),
    ])

SCIENTISTS_GUILD = Card(
    name="Scientists' guild",
    color=Color.PURPLE,
    cost=Cost(resources=[Resource.WOOD, Resource.WOOD, Resource.ORE, Resource.ORE, Resource.PAPYRUS]),
    effects=[
        ScienceEffect([Science.COMPASS, Science.COG, Science.TABLET]),
    ])

MAGISTRATES_GUILD = Card(
    name="Magistrates' guild",
    color=Color.PURPLE,
    cost=Cost(resources=[Resource.WOOD, Resource.WOOD, Resource.WOOD, Resource.STONE, Resource.CLOTH]),
    effects=[
        ScoreEffect(CardCounter(countNeighbors=True, color=Color.BLUE)),
    ])

BUILDERS_GUILD = Card(
    name="Builders' guild",
    color=Color.PURPLE,
    cost=Cost(resources=[Resource.STONE, Resource.STONE, Resource.CLAY, Resource.CLAY, Resource.GLASS]),
    effects=[
        ScoreEffect(WonderCounter(countSelf=True, countNeighbors=True)),
    ])


ALL_CARDS = [
    ACADEMY,
    ALTAR,
    APOTHECARY,
    AQUEDUCT,
    ARCHERY_RANGE,
    ARENA,
    ARSENAL,
    BARRACKS,
    BATHS,
    BAZAR,
    BRICKYARD,
    BUILDERS_GUILD,
    CARAVANSERY,
    # CHAMBER_OF_COMMERCE,
    # CIRCUS,
    CLAY_PIT,
    CLAY_POOL,
    COURTHOUSE,
    CRAFTSMENS_GUILD,
    DISPENSARY,
    EAST_TRADING_POST,
    # EXCAVATION,
    # FOREST_CAVE,
    FORTIFICATIONS,
    FOUNDRY,
    FORUM,
    GARDENS,
    GLASSWORKS,
    GUARD_TOWER,
    HAVEN,
    LABORATORY,
    LIBRARY,
    LIGHTHOUSE,
    LODGE,
    LOOM,
    LUMBER_YARD,
    MAGISTRATES_GUILD,
    MARKETPLACE,
    # MINE,
    OBSERVATORY,
    ORE_VEIN,
    PALACE,
    PANTHEON,
    PAWNSHOP,
    PHILOSOPHERS_GUILD,
    PRESS,
    QUARRY,
    SAWMILL,
    SCHOOL,
    SCIENTISTS_GUILD,
    SCRIPTORIUM,
    SENATE,
    SHIPOWNERS_GUILD,
    SIEGE_WORKSHOP,
    SPIES_GUILD,
    STABLES,
    STATUE,
    STOCKADE,
    STONE_PIT,
    STRATEGISTS_GUILD,
    STUDY,
    TAVERN,
    TEMPLE,
    THEATER,
    TIMBER_YARD,
    TOWN_HALL,
    TRADERS_GUILD,
    # TRAINING_GROUND,
    # TREE_FARM,
    UNIVERSITY,
    VINEYARD,
    WALLS,
    WEST_TRADING_POST,
    WORKERS_GUILD,
    WORKSHOP,
]


def getCardIndex(card):
    for i in range(len(ALL_CARDS)):
        if card.name == ALL_CARDS[i].name:
            return i
    assert(False)


PURPLE_CARDS = [card for card in ALL_CARDS if card.color == Color.PURPLE]

AGE1_3PLAYERS = [
    LUMBER_YARD,
    STONE_PIT,
    CLAY_POOL,
    ORE_VEIN,
    CLAY_PIT,
    TIMBER_YARD,
    LOOM,
    GLASSWORKS,
    PRESS,
    BATHS,
    ALTAR,
    THEATER,
    EAST_TRADING_POST,
    WEST_TRADING_POST,
    MARKETPLACE,
    APOTHECARY,
    WORKSHOP,
    SCRIPTORIUM,
    STOCKADE,
    BARRACKS,
    GUARD_TOWER,
]

AGE2_3PLAYERS = [
    SAWMILL,
    QUARRY,
    BRICKYARD,
    FOUNDRY,
    LOOM,
    GLASSWORKS,
    PRESS,
    AQUEDUCT,
    TEMPLE,
    STATUE,
    FORUM,
    CARAVANSERY,
    VINEYARD,
    DISPENSARY,
    LABORATORY,
    LIBRARY,
    SCHOOL,
    COURTHOUSE,
    WALLS,
    STABLES,
    ARCHERY_RANGE,
]

AGE3_3PLAYERS = [
    PANTHEON,
    GARDENS,
    TOWN_HALL,
    PALACE,
    HAVEN,
    LIGHTHOUSE,
    LODGE,
    OBSERVATORY,
    UNIVERSITY,
    ACADEMY,
    STUDY,
    SENATE,
    FORTIFICATIONS,
    ARSENAL,
    SIEGE_WORKSHOP,
    ARENA,
]

RHODES_A = Wonder(
    name='Rhodes A',
    shortName = 'Rhodes',
    effect=ProductionEffect([Resource.ORE]),
    stages=[
        WonderStage(
            cost=Cost(resources=[Resource.WOOD, Resource.WOOD]),
            effects=[ScoreEffect(Constant(value=3))]),
        WonderStage(
            cost=Cost(resources=[Resource.CLAY, Resource.CLAY, Resource.CLAY]),
            effects=[MilitaryEffect(2)]),
        WonderStage(
            cost=Cost(resources=[Resource.ORE, Resource.ORE, Resource.ORE, Resource.ORE]),
            effects=[ScoreEffect(Constant(value=7))]),
    ])
EPHESUS_A = Wonder(
    name='Ephesus A',
    shortName = 'Ephesus',
    effect=ProductionEffect([Resource.PAPYRUS]),
    stages=[
        WonderStage(
            cost=Cost(resources=[Resource.STONE, Resource.STONE]),
            effects=[ScoreEffect(Constant(value=3))]),
        WonderStage(
            cost=Cost(resources=[Resource.WOOD, Resource.WOOD]),
            effects=[GoldEffect(Constant(value=9))]),
        WonderStage(
            cost=Cost(resources=[Resource.PAPYRUS, Resource.PAPYRUS]),
            effects=[ScoreEffect(Constant(value=7))]),
    ])
GIZA_A = Wonder(
    name='Giza A',
    shortName = 'Giza',
    effect=ProductionEffect([Resource.STONE]),
    stages=[
        WonderStage(
            cost=Cost(resources=[Resource.STONE, Resource.STONE]),
            effects=[ScoreEffect(Constant(value=3))]),
        WonderStage(
            cost=Cost(resources=[Resource.WOOD, Resource.WOOD, Resource.WOOD]),
            effects=[ScoreEffect(Constant(value=5))]),
        WonderStage(
            cost=Cost(resources=[Resource.STONE, Resource.STONE, Resource.STONE, Resource.STONE]),
            effects=[ScoreEffect(Constant(value=7))]),
    ])

ALL_WONDERS = [RHODES_A, EPHESUS_A, GIZA_A]
