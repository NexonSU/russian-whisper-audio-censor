import logging
import os
import sys
import threading
import tqdm
import whisper
import whisper.transcribe 
import io
import re
from pydub import AudioSegment
from pydub.generators import Sine
from tkinter import *
from tkinter import ttk
from tkinter import filedialog

wavfile = ""
audio_result = 0

class PrintLogger(): # create file like object
    def __init__(self, textbox): # pass reference to text widget
        self.textbox = textbox # keep ref

    def write(self, text):
        self.textbox.configure(state=NORMAL)
        self.textbox.insert(END, text)
        self.textbox.configure(state=DISABLED)
        self.textbox.see(END)

    def flush(self): # needed for file like object
        pass

class _CustomProgressBar(tqdm.tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current = self.n
        
    def update(self, n):
        super().update(n)
        self._current += n
        if process_progressbar["value"] == 0:
            logging.info("Распознание речи...")
        process_progressbar["value"] = round(self._current/self.total*100)

def select_file():
    global wavfile

    filetypes = (
        ('Waveform Audio File Format', '*.wav'),
        ('All files', '*.*')
    )

    filename = filedialog.askopenfilename(
        title='Открыть файл',
        initialdir='/',
        filetypes=filetypes)

    if filename != "":
        process_button.configure(state=NORMAL)

        source_field.configure(state=NORMAL)
        source_field.delete(0,END)
        source_field.insert(0,filename)
        source_field.configure(state=DISABLED)

        wavfile = filename

def transcribe():
    global audio_result
    modelname = whisper.load_model(model.get())
    result = modelname.transcribe(audio=wavfile, word_timestamps=True, language=language.get())

    a = AudioSegment.from_wav(wavfile)
    parts = []
    begin = 0
    #regex = re.compile('[\W+]')
    filter = re.compile('(?<![а-яё])(?:(?:(?:у|[нз]а|(?:хитро|не)?вз?[ыьъ]|с[ьъ]|(?:и|ра)[зс]ъ?|(?:о[тб]|п[оа]д)[ьъ]?|(?:\S(?=[а-яё]))+?[оаеи-])-?)?(?:[её](?:б(?!о[рй]|рач)|п[уа](?:ц|тс))|и[пб][ае][тцд][ьъ]).*?|(?:(?:н[иеа]|ра[зс]|[зд]?[ао](?:т|дн[оа])?|с(?:м[еи])?|а[пб]ч)-?)?ху(?:[яйиеёю]|л+и(?!ган)).*?|бл(?:[эя]|еа?)(?:[дт][ьъ]?)?|\S*?(?:п(?:[иеё]зд|ид[аое]?р|ед(?:р(?!о)|[аое]р|ик))|бля(?:[дбц]|тс)|[ое]ху[яйиеёю]|хуйн).*?|(?:о[тб]?|про|на|вы)?м(?:анд(?:[ауеыи](?:л(?:и[сзщ])?[ауеиы])?|ой|[ао]в.*?|юк(?:ов|[ауи])?|е[нт]ь|ища)|уд(?:[яаиое].+?|е?н(?:[ьюия]|ей))|[ао]л[ао]ф[ьъ](?:[яиюе]|[еёо]й))|елд[ауые].*?|ля[тд]ь|(?:[нз]а|по)х)(?![а-яё])')
    #word_filter = io.open("word_filter.txt", mode="r", encoding="utf-8").read().splitlines()

    for segment in result["segments"]:
        for word in segment["words"]:
            #word["word"] = regex.sub('', word["word"]).lower()
            logging.debug("Проверяем " + word["word"] + " на " + str(word["start"]))
            #if word["word"] in word_filter:
            if filter.findall(word["word"].lower()):
                logging.info("Слово " + str(word["word"]) + " обнаружено на " + str(word["start"]))

                s = a[begin*1000:word["start"]*1000]
                parts.append(s)  
                
                duration = (word["end"] - word["start"]) * 1000
                s = Sine(1000).to_audio_segment(duration).apply_gain(-20)
                parts.append(s)

                begin = word["end"]

    parts.append(a[begin*1000:])
 
    audio_result = sum(parts[1:], parts[0])

    logging.info("Готово.")

    process_button.configure(state=NORMAL)

    save_file()

def save_file():
    f = filedialog.asksaveasfilename(defaultextension=".wav")
    if f is None or f == "":
        return
    audio_result.export(f, format='wav')

def start_transcribe_thread(event):
    logging.info("Whisper запускается...")
    process_progressbar["value"] = 0
    process_button.configure(state=DISABLED, text="Сохранить", command=save_file)
    transcribe_thread = threading.Thread(target=transcribe)
    transcribe_thread.daemon = True
    transcribe_thread.start()

if __name__ == '__main__':
    transcribe_module = sys.modules['whisper.transcribe']
    transcribe_module.tqdm.tqdm = _CustomProgressBar

    gui = Tk()
    gui.title('Whisper Audio Censor')
    gui.geometry('450x250')
    gui.resizable(False, False)

    gui.grid_columnconfigure(0, weight=1)

    source_label = Label(gui, text="Источник", padx=5)
    source_label.grid(row=0, column=0, sticky='e')
    source_field = Entry(gui, state=DISABLED, width=50)
    source_field.grid(row=0, column=1, sticky='we', columnspan=3)
    source_button = Button(gui, text='Обзор', command=select_file, padx=5)
    source_button.grid(row=0, column=4, sticky='w')

    languages = ["Afrikaans","Albanian","Amharic","Arabic","Armenian","Assamese","Azerbaijani","Bashkir","Basque","Belarusian","Bengali","Bosnian","Breton","Bulgarian","Burmese","Castilian","Catalan","Chinese","Croatian","Czech","Danish","Dutch","English","Estonian","Faroese","Finnish","Flemish","French","Galician","Georgian","German","Greek","Gujarati","Haitian","Haitian" "Creole","Hausa","Hawaiian","Hebrew","Hindi","Hungarian","Icelandic","Indonesian","Italian","Japanese","Javanese","Kannada","Kazakh","Khmer","Korean","Lao","Latin","Latvian","Letzeburgesch","Lingala","Lithuanian","Luxembourgish","Macedonian","Malagasy","Malay","Malayalam","Maltese","Maori","Marathi","Moldavian","Moldovan","Mongolian","Myanmar","Nepali","Norwegian","Nynorsk","Occitan","Panjabi","Pashto","Persian","Polish","Portuguese","Punjabi","Pushto","Romanian","Russian","Sanskrit","Serbian","Shona","Sindhi","Sinhala","Sinhalese","Slovak","Slovenian","Somali","Spanish","Sundanese","Swahili","Swedish","Tagalog","Tajik","Tamil","Tatar","Telugu","Thai","Tibetan","Turkish","Turkmen","Ukrainian","Urdu","Uzbek","Valencian","Vietnamese","Welsh","Yiddish","Yoruba"]
    language_label = Label(gui, text="Язык", padx=5, state=DISABLED)
    language_label.grid(row=1, column=0, sticky='e')
    language = StringVar(gui)
    language.set("Русский")
    language_optionmenu = OptionMenu(gui, language, *languages)
    language_optionmenu.grid(row=1, column=1, sticky='w')
    #models = ["tiny.en","tiny","base.en","base","small.en","small","medium.en","medium","large-v1","large-v2","large"]
    models = ["tiny","base","small","medium","large"]
    model_label = Label(gui, text="Модель", padx=5)
    model_label.grid(row=1, column=3, sticky='e')
    model = StringVar(gui)
    model.set("large")
    model_optionmenu = OptionMenu(gui, model, *models)
    model_optionmenu.grid(row=1, column=4, sticky='w')

    process_button = Button(gui, text='Запуск', command=lambda:start_transcribe_thread(None), padx=5, state=DISABLED)
    process_button.grid(row=1, column=2)

    process_progressbar = ttk.Progressbar(gui, orient="horizontal", length=100, value=0)
    process_progressbar.grid(row=4, column=0, columnspan=5, sticky='we')
    process_progressbar.columnconfigure(0, weight=1)
    process_log = Text(gui, wrap="none", height=11, state=DISABLED)
    process_log.grid(row=5, column=0, columnspan=5, sticky='w')
    process_log.columnconfigure(0, weight=1)
    sys.stdout = PrintLogger(process_log)
    sys.stderr = PrintLogger(process_log)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    handler = logging.StreamHandler(PrintLogger(process_log))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    root.addHandler(handler)

    logging.info("Выберите файл, язык и модель")

    gui.mainloop()