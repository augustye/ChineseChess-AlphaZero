from collections import defaultdict
from logging import getLogger

import cchess_alphazero.environment.static_env as senv
from cchess_alphazero.environment.chessboard import Chessboard
from cchess_alphazero.environment.chessman import *
from cchess_alphazero.agent.model import CChessModel
from cchess_alphazero.agent.player import CChessPlayer, VisitState
from cchess_alphazero.agent.api import CChessModelAPI
from cchess_alphazero.config import Config
from cchess_alphazero.environment.env import CChessEnv
from cchess_alphazero.environment.lookup_tables import Winner, ActionLabelsRed, flip_move
from cchess_alphazero.lib.model_helper import load_best_model_weight
from cchess_alphazero.lib.tf_util import set_session_config

logger = getLogger(__name__)

def start(config: Config, human_move_first=True):
    set_session_config(per_process_gpu_memory_fraction=1, allow_growth=True, device_list=config.opts.device_list)
    play = PlayWithHuman(config)
    play.start(human_move_first)

class PlayWithHuman:
    def __init__(self, config: Config):
        self.config = config
        self.env = CChessEnv()
        self.model = None
        self.pipe = None
        self.ai = None
        self.chessmans = None
        self.human_move_first = True

    def load_model(self):
        self.model = CChessModel(self.config)
        if self.config.opts.new or not load_best_model_weight(self.model):
            self.model.build()


    def human_move(self):
        self.env.board.calc_chessmans_moving_list()
        while True:
            possible_move_list = []
            title = "请输入着法: "
            input_move_cn = input(title)
            for name in self.env.board.chessmans_hash:
                chessman = self.env.board.chessmans_hash[name]
                if chessman.is_red != self.env.red_to_move:
                    continue
                name_cn = chessman.name_cn[-5]
                if name_cn in input_move_cn:
                    p1 = chessman.position
                    for p2 in chessman.moving_list:
                        if not self.env.red_to_move:
                            move_cn = self.env.board.make_single_record(8-p1.x, 9-p1.y, 8-p2.x, 9-p2.y)
                        else: 
                            move_cn = self.env.board.make_single_record(p1.x, p1.y, p2.x, p2.y)
                        possible_move_list.append(move_cn)
                        if move_cn == input_move_cn:
                            chessman.move(p2.x, p2.y)
                            self.env.board.print_to_cl()
                            self.env.board.clear_chessmans_moving_list()
                            return
            if len(possible_move_list) == 0:
                print("没有这个棋子")
            else:
                print("着法不对，可能的着法为:\n" + "\n".join(possible_move_list))

    def get_ai_action(self):
        self.ai.search_results = {}
        action, policy = self.ai.action(self.env.get_state(), self.env.num_halfmoves)
        key = self.env.get_state()
        p, v = self.ai.debug[key]
        side = "红方" if self.env.red_to_move else "黑方"
        print(f"{side}局势评估：{v:.3f}")
        print(f'MCTS搜索次数：{self.config.play.simulation_num_per_move}')
        labels = ["着法      ", " 访问计数  ", "  动作价值   ", "  先验概率   "] 
        print(f"{labels[0]}{labels[1]}{labels[2]}{labels[3]}")
        for move, action_state in self.ai.search_results.items():
            move_cn = self.env.board.make_single_record(int(move[0]), int(move[1]), int(move[2]), int(move[3]))
            value1 = f"{move_cn}\t"
            value2 = f"　　{action_state[0]:3d}　　"
            value3 = f"　　{action_state[1]:5.2f}　　"
            value4 = f"　　{action_state[2]:5.2f}　　"
            print(f"{value1}{value2}{value3}{value4}")
        return action

    def start(self, human_first=True):
        self.env.reset()
        self.load_model()
        self.pipe = self.model.get_pipes()
        self.ai = CChessPlayer(self.config, search_tree=defaultdict(VisitState), pipes=self.pipe,
                              enable_resign=True, debugging=True)
        self.human_move_first = human_first

        self.env.board.print_to_cl()
        while not self.env.board.is_end():
            if human_first == self.env.red_to_move:
                action = self.get_ai_action()
                self.human_move()
            else:
                action = self.get_ai_action()
                if not self.env.red_to_move:
                    action = flip_move(action)
                if action is None:
                    print("AI投降了!")
                    break
                self.env.step(action)
                self.env.board.print_to_cl()

        self.ai.close()
        print(f"胜者是 is {self.env.board.winner} !!!")
        self.env.board.print_record()
