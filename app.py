from ui import ui
from agent import SessionState as SessionState
if __name__ == "__main__":
    demo = ui(SessionState, app_name="MAGI", revision=None)
    demo.queue()
    demo.launch()
