import kivy
from kivy.uix.screenmanager import Screen

class MenuScreen(Screen):
    pass
    # def _go_to_hall_of_fame(self):
    #     self.parent.get_screen("HallOfFameScreen").make_editable(False)
    #     self.parent.current = 'HallOfFameScreen'

    # def _go_to_game(self):
    #     self.parent.get_screen("GameScreen").set_scene(number_of_players = int(self.ids.number_of_players.text), screen_size = self.size)