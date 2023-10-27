import logging
import sys
import threading
import time
import torch
import psutil
import faster_whisper
import faster_whisper.transcribe 
import re
from pydub import AudioSegment
from pydub.generators import Sine
from tkinter import *
from tkinter import ttk
from tkinter import filedialog

wavfile = ""
audio_result = AudioSegment.empty()
start_time = 0
loading = False

class PrintLogger(): # create file like object
    def __init__(self, textbox): # pass reference to text widget
        self.textbox = textbox # keep ref

    def write(self, text):
        self.textbox.insert(END, text)
        self.textbox.see(END)

    def flush(self): # needed for file like object
        pass

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
    global audio_result, loading

    loading = True
    start_progressbar_thread(None)

    logging.info("Whisper запускается...")

    process_button.configure(state=DISABLED)
    source_label.configure(state=DISABLED)
    source_button.configure(state=DISABLED)
    model_optionmenu.configure(state=DISABLED)
    model_label.configure(state=DISABLED)

    modelname = faster_whisper.WhisperModel(model.get())
    segments, info = modelname.transcribe(audio=wavfile, word_timestamps=True, language="ru", condition_on_previous_text=False, vad_filter=True, no_speech_threshold=0.5)
    total_duration = round(info.duration, 2)

    loading = False
    if use_cuda:
        logging.info('Используется GPU')
        logging.info('CUDNN VERSION: ' + str(torch.backends.cudnn.version()))
        logging.info('Number CUDA Devices: ' + str(torch.cuda.device_count()))
        logging.info('CUDA Device Name: ' + str(torch.cuda.get_device_name(0)))
        logging.info('CUDA Device Total Memory: ' + str(round(torch.cuda.get_device_properties(0).total_memory/1024/1024/1024, 2)) + "GB")
    else:
        logging.info('Используется CPU')
        logging.info('Number of CPUs: ' + str(torch.cpu.device_count()))
        logging.info('Device Total Memory: ' + str(round(psutil.virtual_memory().total/1024/1024/1024, 2)) + "GB")
    process_progressbar.configure(value=0, mode='determinate')
    style.configure('text.Horizontal.TProgressbar', text='0%', anchor='center')
    start_time = time.time()

    a = AudioSegment.from_wav(wavfile)
    last_segment_end = 0
    regex = re.compile('[\W+]')
    filter = re.compile('(?<![а-яё])(?:(?:(?:у|[нз]а|(?:хитро|не)?вз?[ыьъ]|с[ьъ]|(?:и|ра)[зс]ъ?|(?:о[тб]|п[оа]д)[ьъ]?|(?:\S(?=[а-яё]))+?[оаеи-])-?)?(?:[её](?:б(?!о[рй]|рач)|п[уа](?:ц|тс))|и[пб][ае][тцд][ьъ]).*?|(?:(?:н[иеа]|ра[зс]|[зд]?[ао](?:т|дн[оа])?|с(?:м[еи])?|а[пб]ч)-?)?ху(?:[яйиеёю]|л+и(?!ган)).*?|бл(?:[эя]|еа?)(?:[дт][ьъ]?)?|\S*?(?:п(?:[иеё]зд|ид[аое]?р|ед(?:р(?!о)|[аое]р|ик))|бля(?:[дбц]|тс)|[ое]ху[яйиеёю]|хуйн).*?|(?:о[тб]?|про|на|вы)?м(?:анд(?:[ауеыи](?:л(?:и[сзщ])?[ауеиы])?|ой|[ао]в.*?|юк(?:ов|[ауи])?|е[нт]ь|ища)|уд(?:[яаиое].+?|е?н(?:[ьюия]|ей))|[ао]л[ао]ф[ьъ](?:[яиюе]|[еёо]й))|елд[ауые].*?|ля[тд]ь|(?:[нз]а|по)х)(?![а-яё])')
    #word_filter = io.open("word_filter.txt", mode="r", encoding="utf-8").read().splitlines()

    for segment in segments:
        percent = round(segment.start/total_duration, 2)*100
        process_progressbar.configure(value=percent)
        if segment.start - last_segment_end > 5:
            # s = a[last_segment_end*1000:(last_segment_end+1)*1000]
            # s.fade_out(1000)
            # audio_result = audio_result + s
            # audio_result = audio_result + AudioSegment.silent((segment.start - last_segment_end - 2) * 1000)
            # s = a[(segment.start - 1)*1000:segment.start*1000]
            # s.fade_in(1000)
            # audio_result = audio_result + s
            audio_result = audio_result + a[last_segment_end*1000:segment.start*1000]
        else:
            audio_result = audio_result + a[last_segment_end*1000:segment.start*1000]
        if percent > 0 and percent < 99:
            eta = round(((time.time() - start_time)/percent * (100 - percent)))
            style.configure('text.Horizontal.TProgressbar', text=str(round(percent))+'% ETA: ' + time.strftime('%H:%M:%S', time.gmtime(eta)))
        if percent == 100:
            style.configure('text.Horizontal.TProgressbar', text='99%')
        logging.debug(time.strftime('%H:%M:%S', time.gmtime(round(segment.start, 2))) + " - " + segment.text.removeprefix(" "))
        last_badword_end = segment.start
        for word_obj in segment.words:
            word = regex.sub('', word_obj.word).lower()
            logging.debug("Проверяем " + word + " на " + time.strftime('%H:%M:%S', time.gmtime(round(word_obj.start, 2))) + " с " + str(word_obj.probability))
            #if word["word"] in word_filter:
            if filter.findall(word):
                logging.info("Слово " + str(word) + " обнаружено на " + time.strftime('%H:%M:%S', time.gmtime(round(word_obj.start, 2))))

                s = a[last_badword_end*1000:word_obj.start*1000]
                audio_result = audio_result + s
                
                duration = (word_obj.end - word_obj.start) * 1000
                s = Sine(1000).to_audio_segment(duration).apply_gain(-25)
                audio_result = audio_result + s
                last_badword_end = word_obj.end
        audio_result = audio_result + a[last_badword_end*1000:segment.end*1000]
        last_segment_end = segment.end

    # audio_result = audio_result + a[last_segment_end*1000:(last_segment_end + 2)*1000]
    # audio_result = audio_result + AudioSegment.silent(len(a[(last_segment_end + 2)*1000:]))
    audio_result = audio_result + a[last_segment_end*1000:]

    process_progressbar.configure(value=100)
    style.configure('text.Horizontal.TProgressbar', text='100%')

    logging.info("Готово.")

    process_button.configure(state=NORMAL, text="Сохранить", command=save_file)

    save_file()

def save_file():
    f = filedialog.asksaveasfilename(defaultextension=".wav")
    if f is None or f == "":
        return
    audio_result.export(f, format='wav')

def progressbar_load():
    process_progressbar.configure(mode='indeterminate')
    value = 0
    while loading:
        value = value + 1
        process_progressbar.configure(value=value)
        time.sleep(0.03)

def start_transcribe_thread(event):
    transcribe_thread = threading.Thread(target=transcribe)
    transcribe_thread.daemon = True
    transcribe_thread.start()

progressbar_thread = threading.Thread(target=progressbar_load)
progressbar_thread.daemon = True

def start_progressbar_thread(event):
    progressbar_thread.start()

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()

    gui = Tk()
    gui.title('Whisper Audio Censor')
    gui.geometry('800x590')
    gui.resizable(False, False)

    gui.columnconfigure(tuple(range(60)), weight=1)
    gui.rowconfigure(tuple(range(30)), weight=1)

    source_label = Label(gui, text="Источник", padx=5)
    source_label.grid(row=0, column=0, sticky='e')
    source_field = Entry(gui, state=DISABLED, width=50)
    source_field.grid(row=0, column=1, sticky='we', columnspan=3)
    source_button = Button(gui, text='Обзор', command=select_file, padx=5)
    source_button.grid(row=0, column=4, sticky='w')

    #languages = ["Afrikaans","Albanian","Amharic","Arabic","Armenian","Assamese","Azerbaijani","Bashkir","Basque","Belarusian","Bengali","Bosnian","Breton","Bulgarian","Burmese","Castilian","Catalan","Chinese","Croatian","Czech","Danish","Dutch","English","Estonian","Faroese","Finnish","Flemish","French","Galician","Georgian","German","Greek","Gujarati","Haitian","Haitian" "Creole","Hausa","Hawaiian","Hebrew","Hindi","Hungarian","Icelandic","Indonesian","Italian","Japanese","Javanese","Kannada","Kazakh","Khmer","Korean","Lao","Latin","Latvian","Letzeburgesch","Lingala","Lithuanian","Luxembourgish","Macedonian","Malagasy","Malay","Malayalam","Maltese","Maori","Marathi","Moldavian","Moldovan","Mongolian","Myanmar","Nepali","Norwegian","Nynorsk","Occitan","Panjabi","Pashto","Persian","Polish","Portuguese","Punjabi","Pushto","Romanian","Russian","Sanskrit","Serbian","Shona","Sindhi","Sinhala","Sinhalese","Slovak","Slovenian","Somali","Spanish","Sundanese","Swahili","Swedish","Tagalog","Tajik","Tamil","Tatar","Telugu","Thai","Tibetan","Turkish","Turkmen","Ukrainian","Urdu","Uzbek","Valencian","Vietnamese","Welsh","Yiddish","Yoruba"]
    languages = ["Русский"]
    language_label = Label(gui, text="Язык", padx=5)
    language_label.configure(state=DISABLED)
    language_label.grid(row=1, column=0, sticky='e')
    language = StringVar(gui)
    language.set("Русский")
    language_optionmenu = OptionMenu(gui, language, *languages)
    language_optionmenu.configure(state=DISABLED)
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

    style = ttk.Style(gui)
    style.layout('text.Horizontal.TProgressbar', 
             [('Horizontal.Progressbar.trough',
               {'children': [('Horizontal.Progressbar.pbar',
                              {'side': 'left', 'sticky': 'ns'})],
                'sticky': 'nswe'}), 
              ('Horizontal.Progressbar.label', {'sticky': 'nswe'})])

    process_log = Text(gui, wrap="none", width=2000, height=31)
    xsb = Scrollbar(gui, orient = HORIZONTAL)
    ysb = Scrollbar(gui)
    process_progressbar = ttk.Progressbar(gui, length=2000, orient="horizontal", maximum=100, value=0, style='text.Horizontal.TProgressbar', mode='determinate')
    process_progressbar.grid(row=4, column=0, columnspan=5, sticky=E+W)
    process_progressbar.columnconfigure(0, weight=1)
    ysb.configure(command = process_log.yview)
    ysb.grid(row = 5, column = 4, sticky = "nes")
    ysb.lift()
    xsb.configure(command = process_log.xview)
    xsb.grid(row = 6, column = 0, columnspan=5, sticky = "new")
    xsb.lift()
    process_log.configure(yscrollcommand = ysb.set, xscrollcommand = xsb.set)
    process_log.grid(row=5, column=0, columnspan=5, sticky = "news")

    # process_log = Text(gui, wrap="none", height=2000, width=2000, state=DISABLED)
    # process_log.grid(row=5, column=0, columnspan=5, sticky='w')
    # process_log.columnconfigure(0, weight=1)

    sys.stdout = PrintLogger(process_log)
    sys.stderr = PrintLogger(process_log)
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    handler = logging.StreamHandler(PrintLogger(process_log))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    root.addHandler(handler)

    logging.info("Выберите файл, язык и модель")

    gui.mainloop()