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

    def start(self, human_first=True):
        self.env.reset()
        self.load_model()
        self.pipe = self.model.get_pipes()
        self.ai = CChessPlayer(self.config, search_tree=defaultdict(VisitState), pipes=self.pipe,
                              enable_resign=True, debugging=True)
        self.human_move_first = human_first

        labels = ActionLabelsRed
        labels_n = len(ActionLabelsRed)

        self.env.board.print_to_cl()

        while not self.env.board.is_end():
            if human_first == self.env.red_to_move:
                self.env.board.calc_chessmans_moving_list()
                is_correct_chessman = False
                is_correct_position = False
                chessman = None
                while not is_correct_chessman:
                    title = "请输入棋子位置: "
                    input_chessman_pos = input(title)
                    try:
                        x, y = "ABCDEFGHI".index(input_chessman_pos[0]), int(input_chessman_pos[1])
                        chessman = self.env.board.chessmans[x][y]
                    except:
                        pass
                    if chessman != None and chessman.is_red == self.env.board.is_red_turn:
                        is_correct_chessman = True
                        print(f"当前棋子为{chessman.name_cn}")
                        #for point in chessman.moving_list:
                        #    print(point.x, point.y)
                    else:
                        print("没有找到此名字的棋子或未轮到此方走子")
                while not is_correct_position:
                    title = "请输入落子的位置: "
                    input_chessman_pos = input(title)
                    try:
                        x, y = "ABCDEFGHI".index(input_chessman_pos[0]), int(input_chessman_pos[1])
                        is_correct_position = chessman.move(x, y)
                    except:
                        pass
                    if is_correct_position:
                        self.env.board.print_to_cl()
                        self.env.board.clear_chessmans_moving_list()
            else:
                self.ai.search_results = {}
                action, policy = self.ai.action(self.env.get_state(), self.env.num_halfmoves)
                if not self.env.red_to_move:
                    action = flip_move(action)
                if action is None:
                    print("AI投降了!")
                    break
                x_list = "ABCDEFGHI"
                key = self.env.get_state()
                p, v = self.ai.debug[key]
                chessman = self.env.board.chessmans[int(action[0])][int(action[1])]
                print(f"当前局势评估：{v:.3f}")
                print(f'MCTS搜索次数：{self.config.play.simulation_num_per_move}')
                print(f"AI选择移动{chessman.name_cn}：{x_list[int(action[0])]+action[1]} -> {x_list[int(action[2])]+action[3]}\n")
                labels = ["     着法    ", " 访问计数  ", "  动作价值   ", "  先验概率   "] 
                print(f"{labels[0]}{labels[1]}{labels[2]}{labels[3]}")
                for move, action_state in self.ai.search_results.items():
                    if not self.env.red_to_move:
                        move = flip_move(move)
                    chessman = self.env.board.chessmans[int(move[0])][int(move[1])]
                    value1 = f"{chessman.name_cn}　{x_list[int(move[0])]}{int(move[1])} -> {x_list[int(move[2])]}{int(move[3])} "
                    value2 = f"　　{action_state[0]:3d}　　"
                    value3 = f"　　{action_state[1]:5.2f}　　"
                    value4 = f"　　{action_state[2]:5.2f}　　"
                    print(f"{value1}{value2}{value3}{value4}")
                self.env.step(action)
                self.env.board.print_to_cl()

        self.ai.close()
        print(f"胜者是 is {self.env.board.winner} !!!")
        self.env.board.print_record()
