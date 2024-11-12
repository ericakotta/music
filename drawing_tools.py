class musicDrawer():
    def __init__(self,
        time_signature_top: int = 4,
        time_signature_bottom: int = 4,
        smallest_beat: float = 0.5,

    ):
        # Define the time signature 
        self.time_signature_top = time_signature_top
        self.time_signature_bottom = time_signature_bottom

        # Define the smallest beat resolved by the music (0.5 is eigth-note, etc)
        self.beat_unit = smallest_beat
        self.num_beats = int(self.time_signature_bottom / self.beat_unit)
        self.num_lines_btw_clefs = 6

        treble_lines = self.draw_five_lines()
        bass_lines = self.draw_five_lines()
        middle_lines = '\n' * self.num_lines_btw_clefs
        measure = treble_lines + middle_lines + bass_lines
        measure = self.attach_first_measure_prefix(measure)
        
        measure_lines = self.draw_measure()
        print('\n'.join(measure_lines))

        measure_lines = self.attach_first_measure_prefix(measure_lines)
        print('\n'.join(measure_lines))

    def draw_measure(self, num_beats: int = None):
        # Optionally specify number of beats. Defaults to full measure
        if num_beats is None:
            num_beats = self.num_beats
        line = '-' + '-' * (2 * num_beats) + '-'
        empty_line = ' ' + ' ' * (2 * num_beats) + ' '
        treble_lines = [line] * 5
        bass_lines = [line] * 5
        middle_lines = [empty_line] * 5
        return treble_lines + middle_lines + bass_lines
    
    def stitch_measures(self, measures_list: list):
        return

    def draw_five_lines(self):
        line =  '-' * (2 * self.num_beats) + '-'
        return '\n'.join([line] * 5)
    
    def attach_first_measure_prefix(self, measure_lines: list):
        new_measure_lines = []
        for line in measure_lines:
            new_measure_lines.append('|' + line)
        return new_measure_lines
    
    def draw_measure_notes(self, notes_dict: dict):
        """Input: notes_dict dictionary of notes to draw"""

if __name__ == '__main__':
    musicDrawer()