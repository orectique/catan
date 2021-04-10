from tests.utils import advance_to_play_turn, build_initial_placements, end_turn
from catanatron.models.map import BaseMap
from catanatron.models.board import Board
from catanatron.game import Game
from catanatron.algorithms import longest_road, largest_army
from catanatron.models.actions import Action, ActionType
from catanatron.models.player import SimplePlayer, Color
from catanatron.models.decks import ResourceDeck
from catanatron.models.enums import DevelopmentCard, Resource


def test_longest_road_simple():
    red = SimplePlayer(Color.RED)
    blue = SimplePlayer(Color.BLUE)
    red.resource_deck += ResourceDeck.starting_bank()
    blue.resource_deck += ResourceDeck.starting_bank()

    game = Game(players=[red, blue])
    build_initial_placements(game, [0, (0, 1), 2, (1, 2)])
    advance_to_play_turn(game)

    color, path = longest_road(game.state.board, game.state.players, game.state.actions)
    assert color is None

    p0_color = game.state.players[0].color
    game.state.players[0].resource_deck.replenish(10, Resource.WOOD)
    game.state.players[0].resource_deck.replenish(10, Resource.BRICK)
    game.execute(Action(p0_color, ActionType.BUILD_ROAD, (2, 3)))
    game.execute(Action(p0_color, ActionType.BUILD_ROAD, (3, 4)))
    game.execute(Action(p0_color, ActionType.BUILD_ROAD, (4, 5)))

    color, path = longest_road(game.state.board, game.state.players, game.state.actions)
    assert color == p0_color
    assert len(path) == 5


def test_longest_road_tie():
    red = SimplePlayer(Color.RED)
    blue = SimplePlayer(Color.BLUE)
    red.resource_deck += ResourceDeck.starting_bank()
    blue.resource_deck += ResourceDeck.starting_bank()

    game = Game(players=[red, blue])
    build_initial_placements(game, [0, (0, 1), 2, (1, 2)])
    advance_to_play_turn(game)

    p0_color = game.state.players[0].color
    game.state.players[0].resource_deck.replenish(10, Resource.WOOD)
    game.state.players[0].resource_deck.replenish(10, Resource.BRICK)
    game.execute(Action(p0_color, ActionType.BUILD_ROAD, (2, 3)))
    game.execute(Action(p0_color, ActionType.BUILD_ROAD, (3, 4)))
    game.execute(Action(p0_color, ActionType.BUILD_ROAD, (4, 5)))
    end_turn(game)
    advance_to_play_turn(game)

    p1_color = game.state.players[1].color
    game.state.players[1].resource_deck.replenish(10, Resource.WOOD)
    game.state.players[1].resource_deck.replenish(10, Resource.BRICK)
    game.execute(Action(p1_color, ActionType.BUILD_ROAD, (26, 27)))
    game.execute(Action(p1_color, ActionType.BUILD_ROAD, (27, 28)))
    game.execute(Action(p1_color, ActionType.BUILD_ROAD, (28, 29)))

    color, path = longest_road(game.state.board, game.state.players, game.state.actions)
    assert color == p0_color  # even if blue also has 5-road. red had it first
    assert len(path) == 5

    game.execute(Action(p1_color, ActionType.BUILD_ROAD, (29, 30)))
    color, path = longest_road(game.state.board, game.state.players, game.state.actions)
    assert color == p1_color
    assert len(path) == 6


# test: complicated circle around
def test_complicated_road():  # classic 8-like roads
    red = SimplePlayer(Color.RED)
    blue = SimplePlayer(Color.BLUE)
    red.resource_deck += ResourceDeck.starting_bank()
    blue.resource_deck += ResourceDeck.starting_bank()

    game = Game(players=[red, blue])
    build_initial_placements(game, [0, (0, 1), 2, (1, 2)])
    advance_to_play_turn(game)

    p0_color = game.state.players[0].color
    game.state.players[0].resource_deck.replenish(20, Resource.WOOD)
    game.state.players[0].resource_deck.replenish(20, Resource.BRICK)
    game.execute(Action(p0_color, ActionType.BUILD_ROAD, (2, 3)))
    game.execute(Action(p0_color, ActionType.BUILD_ROAD, (3, 4)))
    game.execute(Action(p0_color, ActionType.BUILD_ROAD, (4, 5)))
    game.execute(Action(p0_color, ActionType.BUILD_ROAD, (0, 5)))

    game.execute(Action(p0_color, ActionType.BUILD_ROAD, (1, 6)))
    game.execute(Action(p0_color, ActionType.BUILD_ROAD, (6, 7)))
    game.execute(Action(p0_color, ActionType.BUILD_ROAD, (7, 8)))
    game.execute(Action(p0_color, ActionType.BUILD_ROAD, (8, 9)))
    game.execute(Action(p0_color, ActionType.BUILD_ROAD, (2, 9)))

    color, path = longest_road(game.state.board, game.state.players, game.state.actions)
    assert color == p0_color
    assert len(path) == 11

    game.execute(Action(p0_color, ActionType.BUILD_ROAD, (8, 27)))

    color, path = longest_road(game.state.board, game.state.players, game.state.actions)
    assert color == p0_color
    assert len(path) == 11


def test_triple_longest_road_tie():
    red = SimplePlayer(Color.RED)
    blue = SimplePlayer(Color.BLUE)
    white = SimplePlayer(Color.WHITE)
    red.resource_deck += ResourceDeck.starting_bank()
    blue.resource_deck += ResourceDeck.starting_bank()
    white.resource_deck += ResourceDeck.starting_bank()

    board = Board(BaseMap())
    board.build_settlement(Color.RED, 3, True)
    board.build_road(Color.RED, (3, 2))
    board.build_road(Color.RED, (2, 1))
    board.build_road(Color.RED, (1, 0))
    board.build_road(Color.RED, (0, 5))
    board.build_road(Color.RED, (5, 4))
    board.build_road(Color.RED, (3, 4))

    board.build_settlement(Color.BLUE, 24, True)
    board.build_road(Color.BLUE, (24, 25))
    board.build_road(Color.BLUE, (25, 26))
    board.build_road(Color.BLUE, (26, 27))
    board.build_road(Color.BLUE, (27, 8))
    board.build_road(Color.BLUE, (8, 7))
    board.build_road(Color.BLUE, (7, 24))

    board.build_settlement(Color.WHITE, 17, True)
    board.build_road(Color.WHITE, (18, 17))
    board.build_road(Color.WHITE, (17, 39))
    board.build_road(Color.WHITE, (39, 41))
    board.build_road(Color.WHITE, (41, 42))
    board.build_road(Color.WHITE, (42, 40))
    board.build_road(Color.WHITE, (40, 18))
    board.update_connected_components()

    # subset... not quite representative, but should be ok.
    actions = [
        Action(Color.RED, ActionType.BUILD_ROAD, (3,2)),
        Action(Color.BLUE, ActionType.BUILD_ROAD, (24,25)),
        Action(Color.WHITE, ActionType.BUILD_ROAD, (18,17)),
    ]

    color, path = longest_road(board, [red, blue, white], actions)
    assert color == Color.RED
    assert len(path) == 6


def test_largest_army_calculation_when_no_one_has_three():
    red = SimplePlayer(Color.RED)
    blue = SimplePlayer(Color.BLUE)
    white = SimplePlayer(Color.WHITE)

    red.development_deck.replenish(2, DevelopmentCard.KNIGHT)
    blue.development_deck.replenish(1, DevelopmentCard.KNIGHT)
    red.mark_played_dev_card(DevelopmentCard.KNIGHT)
    red.mark_played_dev_card(DevelopmentCard.KNIGHT)
    blue.mark_played_dev_card(DevelopmentCard.KNIGHT)
    actions = [
        Action(Color.RED, ActionType.PLAY_KNIGHT_CARD, None),
        Action(Color.RED, ActionType.PLAY_KNIGHT_CARD, None),
        Action(Color.BLUE, ActionType.PLAY_KNIGHT_CARD, None),
    ]

    color, count = largest_army([red, blue, white], actions)
    assert color is None and count is None


def test_largest_army_calculation_on_tie():
    red = SimplePlayer(Color.RED)
    blue = SimplePlayer(Color.BLUE)
    white = SimplePlayer(Color.WHITE)

    red.development_deck.replenish(3, DevelopmentCard.KNIGHT)
    blue.development_deck.replenish(4, DevelopmentCard.KNIGHT)
    red.mark_played_dev_card(DevelopmentCard.KNIGHT)
    red.mark_played_dev_card(DevelopmentCard.KNIGHT)
    red.mark_played_dev_card(DevelopmentCard.KNIGHT)
    blue.mark_played_dev_card(DevelopmentCard.KNIGHT)
    blue.mark_played_dev_card(DevelopmentCard.KNIGHT)
    blue.mark_played_dev_card(DevelopmentCard.KNIGHT)
    actions = [
        Action(Color.RED, ActionType.PLAY_KNIGHT_CARD, None),
        Action(Color.RED, ActionType.PLAY_KNIGHT_CARD, None),
        Action(Color.RED, ActionType.PLAY_KNIGHT_CARD, None),
        Action(Color.BLUE, ActionType.PLAY_KNIGHT_CARD, None),
        Action(Color.BLUE, ActionType.PLAY_KNIGHT_CARD, None),
        Action(Color.BLUE, ActionType.PLAY_KNIGHT_CARD, None),
    ]

    color, count = largest_army([red, blue, white], actions)
    assert color is Color.RED and count == 3

    blue.mark_played_dev_card(DevelopmentCard.KNIGHT)
    actions.append(Action(Color.BLUE, ActionType.PLAY_KNIGHT_CARD, None))

    color, count = largest_army([red, blue, white], actions)
    assert color is Color.BLUE and count == 4


def test_cut_but_not_disconnected():
    board = Board()

    board.build_settlement(Color.RED, 0, initial_build_phase=True)
    board.build_road(Color.RED, (0, 1))
    board.build_road(Color.RED, (1, 2))
    board.build_road(Color.RED, (2, 3))
    board.build_road(Color.RED, (3, 4))
    board.build_road(Color.RED, (4, 5))
    board.build_road(Color.RED, (5, 0))
    board.build_road(Color.RED, (3, 12))
    assert (
        max(map(lambda path: len(path), board.continuous_roads_by_player(Color.RED)))
        == 7
    )
    assert len(board.find_connected_components(Color.RED)) == 1

    board.build_settlement(Color.BLUE, 2, initial_build_phase=True)
    assert len(board.find_connected_components(Color.RED)) == 1
    assert (
        max(map(lambda path: len(path), board.continuous_roads_by_player(Color.RED)))
        == 6
    )
