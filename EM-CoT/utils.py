import numpy as np
import random
import torch
import signal
import functools
import sys
sys.path.append('/scratch/jiarui14/EM-CoT/Online-DPO-R1')
import reward_labeling

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class TimeoutError(Exception):
    """超时异常"""
    pass

def timeout(seconds, error_message='函数执行超时'):
    """
    装饰器：如果被装饰的函数执行时间超过指定秒数，则抛出 TimeoutError 异常。
    
    参数:
      seconds (int): 允许的最大执行时间（秒）。
      error_message (str): 超时后抛出的异常信息。
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def _handle_timeout(signum, frame):
                raise TimeoutError(error_message)
            # 设置超时信号处理函数
            old_handler = signal.signal(signal.SIGALRM, _handle_timeout)
            # 启动计时器
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                # 取消计时器，并恢复之前的信号处理函数
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            return result
        return wrapper
    return decorator

@timeout(10)
def check_correct(output, answer, idx, threshold=-1.0, accept_rates=None):
    if reward_labeling.is_equal(output, answer, dataset_name='math500'):
        reward = 1.0
    elif "\\boxed" in output:
        reward = -0.5
    else:
        reward = -1.0

    if reward > threshold:
        if not accept_rates:
            return True
        elif np.random.rand() < accept_rates[idx]:
            return True
        
    return False

if __name__ == "__main__":
    # 示例函数，执行时间超过超时时间
    @timeout(3)
    def long_running_function():
        import time
        print("函数开始执行")
        time.sleep(2)  # 模拟长时间运行
        return "执行完成"

    try:
        print(check_correct("To solve this problem, we need to find the points of intersection of the two circles and then determine the slope of the line passing through these points.\n\n1. **Equations of the Circles:**\n   - The first circle has a radius of 4 and is centered at \\((4, 0)\\). Its equation is \\((x - 4)^2 + y^2 = 16\\).\n   - The second circle has a radius of 10 and is centered at \\((0, 10)\\). Its equation is \\(x^2 + (y - 10)^2 = 100\\).\n\n2. **Finding the Points of Intersection:**\n   We need to solve the system of equations:\n   \\[\n   \\begin{cases}\n   (x - 4)^2 + y^2 = 16 \\\\\n   x^2 + (y - 10)^2 = 100\n   \\end{cases}\n   \\]\n\n3. **Solving the System of Equations:**\n   Let's expand and simplify the equations:\n   \\[\n   (x - 4)^2 + y^2 = 16 \\implies x^2 - 8x + 16 + y^2 = 16 \\implies x^2 - 8x + y^2 = 0 \\quad \\text{(Equation 1)}\n   \\]\n   \\[\n   x^2 + (y - 10)^2 = 100 \\implies x^2 + y^2 - 20y + 100 = 100 \\implies x^2 + y^2 - 20y = 0 \\quad \\text{(Equation 2)}\n   \\]\n\n   Subtract Equation 1 from Equation 2:\n   \\[\n   (x^2 + y^2 - 20y) - (x^2 - 8x + y^2) = 0 \\implies 8x - 20y = 0 \\implies 2x = 5y \\implies x = \\frac{5y}{2}\n   \\]\n\n   Substitute \\(x = \\frac{5y}{2}\\) into Equation 1:\n   \\[\n   \\left(\\frac{5y}{2}\\right)^2 - 8\\left(\\frac{5y}{2}\\right) + y^2 = 0 \\implies \\frac{25y^2}{4} - 20y + y^2 = 0 \\implies \\frac{25y^2 + 4y^2 - 80y}{4} = 0 \\implies 29y^2 - 80y = 0 \\implies y(29y - 80) = 0\n   \\]\n\n   So, \\(y = 0\\) or \\(y = \\frac{80}{29}\\). Since \\(y = 0\\) corresponds to the origin, we take \\(y = \\frac{80}{29}\\). Then:\n   \\[\n   x = \\frac{5 \\cdot \\frac{80}{29}}{2} = \\frac{200}{29}\n   \\]\n\n   So, the points of intersection are \\((0, 0)\\) and \\(\\left(\\frac{200}{29}, \\frac{80}{29}\\right)\\).\n\n4. **Finding the Slope:**\n   The slope \\(m\\) of the line passing through \\((0, 0)\\) and \\(\\left(\\frac{200}{29}, \\frac{80}{29}\\right)\\) is:\n   \\[\n   m = \\frac{\\frac{80}{29} - 0}{\\frac{200}{29} - 0} = \\frac{80}{200} = \\frac{2}{5}\n   \\]\n\n   So, the slope is \\(\\frac{2}{5}\\), and \\(m = 2\\) and \\(n = 5\\). Therefore, \\(m + n = 2 + 5 = 7\\).\n\nLet's confirm this with Python code.\n```python\nfrom sympy import symbols, Eq, solve\r\n\r\n# Define the variables\r\nx, y = symbols('x y')\r\n\r\n# Define the equations of the circles\r\neq1 = Eq((x - 4)**2 + y**2, 16)\r\neq2 = Eq(x**2 + (y - 10)**2, 100)\r\n\r\n# Solve the system of equations\r\nsolutions = solve((eq1, eq2), (x, y))\r\nprint(solutions)\n```\n```output\n[(0, 0), (200/29, 80/29)]\n```\nThe solutions to the system of equations are \\((0, 0)\\) and \\(\\left(\\frac{200}{29}, \\frac{80}{29}\\right)\\), confirming our earlier calculations. The slope of the line passing through these points is \\(\\frac{2}{5}\\), and thus \\(m = 2\\) and \\(n = 5\\). Therefore, \\(m + n = 2 + 5 = 7\\).\n\nThe final answer is \\(\\boxed{7}\\).", "7", 0))
    except TimeoutError as e:
        print(f"捕获超时异常: {e}")
