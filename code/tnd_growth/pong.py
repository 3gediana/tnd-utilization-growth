"""Pong 评测环境（纯 Python，无 pygame/screen 依赖，逐位复刻上游 pygame 版本的游戏逻辑）。

上游用 pygame.Rect 表示挡板/球，用 random.choice 放球。这里用等价的 Rect 类与同一
随机调用顺序（random.seed(episode_seed) 后依次取球初始 x、y，每次失球再取一次 x、y），
因此同一模型 + 同一 episode seed 的得分与上游一致。

一局 = 2000 步（论文协议）。catch_rate = 接住次数 / (接住 + 失球)。
"""
from __future__ import annotations

import random
from typing import Any

import torch

SCREEN_WIDTH, SCREEN_HEIGHT = 200, 300
PADDLE_WIDTH, PADDLE_HEIGHT, PADDLE_SPEED = 50, 10, 7
BALL_SIZE = 15
BALL_SPEED_X, BALL_SPEED_Y = 4, -4
MAX_STEPS = 2000


class Rect:
    __slots__ = ("x", "y", "w", "h")

    def __init__(self, x: int, y: int, w: int, h: int):
        self.x, self.y, self.w, self.h = int(x), int(y), int(w), int(h)

    @property
    def left(self) -> int:
        return self.x

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def top(self) -> int:
        return self.y

    @property
    def bottom(self) -> int:
        return self.y + self.h

    def colliderect(self, o: "Rect") -> bool:
        # pygame 语义：严格重叠（接触边不算碰撞）
        return (self.left < o.right and self.right > o.left
                and self.top < o.bottom and self.bottom > o.top)


def int_to_binary_list(n: int, bits: int = 8) -> list[int]:
    """与上游 fun.int_to_binary_list 一致：n 的 bits 位补码二进制。"""
    twos = n & (2 ** bits - 1)
    return [int(b) for b in format(twos, f"0{bits}b")]


def _obs(ball: Rect, paddle: Rect) -> torch.Tensor:
    tmp = []
    tmp.extend(int_to_binary_list(int(ball.x) + 2))
    tmp.extend(int_to_binary_list(int(ball.y) + 2))
    tmp.extend(int_to_binary_list(int(paddle.x) + 2))
    return torch.tensor(tmp, dtype=torch.float32).reshape(1, -1, 24)


def play_episode(model, episode_seed: int, max_steps: int = MAX_STEPS,
                 device: str = "cpu") -> dict[str, Any]:
    random.seed(int(episode_seed))
    torch.manual_seed(int(episode_seed))

    paddle = Rect(SCREEN_WIDTH // 2 - PADDLE_WIDTH // 2, SCREEN_HEIGHT - 40,
                  PADDLE_WIDTH, PADDLE_HEIGHT)
    ball = Rect(random.choice([i * 10 for i in range(3, 17)]),
                random.choice([i * 10 for i in range(3, 25)]), BALL_SIZE, BALL_SIZE)
    ball_dx, ball_dy = BALL_SPEED_X, BALL_SPEED_Y

    states = None
    neuron_outputs = None
    succ = fail = trail = best_trail = 0

    with torch.no_grad():
        for _ in range(max_steps):
            x_input = _obs(ball, paddle).to(device)
            pred, states, neuron_outputs = model.pred(x_input, states, neuron_outputs)
            ctr = int(torch.argmax(pred).item())
            press = 1 if ctr == 0 else (-1 if ctr == 1 else 0)

            if press == -1 and paddle.left > 0:
                paddle.x -= PADDLE_SPEED
            if press == 1 and paddle.right < SCREEN_WIDTH:
                paddle.x += PADDLE_SPEED

            ball.x += ball_dx
            ball.y += ball_dy

            if ball.left <= 0 or ball.right >= SCREEN_WIDTH:
                ball_dx *= -1
            if ball.top <= 0:
                ball_dy *= -1
            if ball.bottom >= SCREEN_HEIGHT:
                fail += 1
                trail = 0
                ball = Rect(random.choice([i * 10 for i in range(3, 17)]),
                            random.choice([i * 10 for i in range(3, 25)]), BALL_SIZE, BALL_SIZE)
                ball_dx, ball_dy = BALL_SPEED_X, BALL_SPEED_Y

            if ball.colliderect(paddle) and ball_dy > 0:
                ball_dy *= -1
                succ += 1
                trail += 1
                best_trail = max(best_trail, trail)

    total = succ + fail
    return {
        "episode_seed": int(episode_seed),
        "succ": succ,
        "fail": fail,
        "catch_rate": succ / total if total > 0 else 0.0,
        "best_trail": best_trail,
    }


def evaluate(model, episode_seeds: list[int], max_steps: int = MAX_STEPS,
             device: str = "cpu") -> dict[str, Any]:
    model.eval()
    episodes = [play_episode(model, s, max_steps=max_steps, device=device)
                for s in episode_seeds]
    rates = [e["catch_rate"] for e in episodes]
    return {
        "episodes": episodes,
        "mean_catch_rate": sum(rates) / len(rates),
        "mean_best_trail": sum(e["best_trail"] for e in episodes) / len(episodes),
        "num_episodes": len(episodes),
        "max_steps_per_episode": max_steps,
    }