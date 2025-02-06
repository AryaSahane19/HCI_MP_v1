import eel
import os
from queue import Queue
from pathlib import Path

class ChatBot:
    started = False
    userinputQueue = Queue()

    @staticmethod
    def isUserInput():
        return not ChatBot.userinputQueue.empty()

    @staticmethod
    def popUserInput():
        return ChatBot.userinputQueue.get()

    @staticmethod
    def close_callback(route, websockets):
        os._exit(0)

    @eel.expose
    def getUserInput(msg):
        ChatBot.userinputQueue.put(msg)
        print("User input:", msg)
    
    @staticmethod
    def close():
        ChatBot.started = False
    
    @staticmethod
    def addUserMsg(msg):
        eel.addUserMsg(msg)
    
    @staticmethod
    def addAppMsg(msg):
        eel.addAppMsg(msg)

    @staticmethod
    def start():
        path = Path(__file__).parent.absolute()
        web_path = path / 'web'
        eel.init(str(web_path), allowed_extensions=['.js', '.html'])
        
        try:
            eel.start('index.html',
                      mode='default',
                      host='localhost',
                      port=27005,
                      block=False,
                      size=(350, 480),
                      position=(10, 100),
                      disable_cache=True,
                      close_callback=ChatBot.close_callback)
            
            ChatBot.started = True
            while ChatBot.started:
                try:
                    eel.sleep(1.0)
                except (KeyboardInterrupt, SystemExit):
                    break
                except Exception as e:
                    print("Unexpected error:", e)
                    break
        
        except Exception as e:
            print("Failed to start Eel:", e)