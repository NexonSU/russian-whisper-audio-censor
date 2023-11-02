import glob
import io
import logging
import multiprocessing
import os
import sys
import threading
import time
from tkinter import messagebox
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
import sv_ttk
import sys

model = ""
wavfiles = []
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
    global wavfiles

    folder = filedialog.askdirectory(
        title='Укажите папку с wav-файлами',
        initialdir='/')

    if folder != "" and len(glob.glob(folder+'/*.wav')):
        process_button.configure(state=NORMAL)

        source_field.configure(state=NORMAL)
        source_field.delete(0,END)
        source_field.insert(0,folder)
        source_field.configure(state=DISABLED)

        for file in glob.glob(folder+'/*.wav'):
            if "_clean.wav" not in file:
                wavfiles.append(file)
    else:
        messagebox.showwarning(title="Файлы не найдены", message="В указанной папке нет wav-файлов.")

def transcribe():
    global loading

    process_button.configure(state=DISABLED)
    source_label.configure(state=DISABLED)
    source_button.configure(state=DISABLED)

    if torch.cuda.is_available():
        logging.info('Используется GPU')
        logging.info('CUDNN VERSION: ' + str(torch.backends.cudnn.version()))
        logging.info('Number CUDA Devices: ' + str(torch.cuda.device_count()))
        logging.info('CUDA Device Name: ' + str(torch.cuda.get_device_name(0)))
        logging.info('CUDA Device Total Memory: ' + str(round(torch.cuda.get_device_properties(0).total_memory/1024/1024/1024, 2)) + "GB")
    else:
        logging.info('Используется CPU')
        logging.info('Number of CPUs: ' + str(torch.cpu.device_count()))
        logging.info('Device Total Memory: ' + str(round(psutil.virtual_memory().total/1024/1024/1024, 2)) + "GB")

    for wavfile in wavfiles:
        audio_result = AudioSegment.empty()

        start_progressbar_thread()

        logging.info("Начинаю обработку " + os.path.basename(wavfile))

        modelname = faster_whisper.WhisperModel(model.get())
        segments, info = modelname.transcribe(audio=wavfile, word_timestamps=True, language="ru", condition_on_previous_text=False, vad_filter=True, no_speech_threshold=0.5)
        total_duration = round(info.duration, 2)

        loading = False
        process_progressbar.configure(value=0, mode='determinate')
        style.configure('text.Horizontal.TProgressbar', text='0%', anchor='center')
        start_time = time.time()

        a = AudioSegment.from_wav(wavfile)
        last_segment_end = 0
        regex = re.compile('[\W+]')
        filter = re.compile('(?<![а-яё])(?:(?:(?:у|[нз]а|(?:хитро|не)?вз?[ыьъ]|с[ьъ]|(?:и|ра)[зс]ъ?|(?:о[тб]|п[оа]д)[ьъ]?|(?:\S(?=[а-яё]))+?[оаеи-])-?)?(?:[её](?:б(?!о[рй]|рач)|п[уа](?:ц|тс))|и[пб][ае][тцд][ьъ]).*?|(?:(?:н[иеа]|ра[зс]|[зд]?[ао](?:т|дн[оа])?|с(?:м[еи])?|а[пб]ч)-?)?ху(?:[яйиеёю]|л+и(?!ган)).*?|бл(?:[эя]|еа?)(?:[дт][ьъ]?)?|\S*?(?:п(?:[иеё]зд|ид[аое]?р|ед(?:р(?!о)|[аое]р|ик))|бля(?:[дбц]|тс)|[ое]ху[яйиеёю]|хуйн).*?|(?:о[тб]?|про|на|вы)?м(?:анд(?:[ауеыи](?:л(?:и[сзщ])?[ауеиы])?|ой|[ао]в.*?|юк(?:ов|[ауи])?|е[нт]ь|ища)|уд(?:[яаиое].+?|е?н(?:[ьюия]|ей))|[ао]л[ао]ф[ьъ](?:[яиюе]|[еёо]й))|елд[ауые].*?|ля[тд]ь|(?:[нз]а|по)х)(?![а-яё])')
        #word_filter = io.open("word_filter.txt", mode="r", encoding="utf-8").read().splitlines()
        whitelist = io.open("whitelist.txt", mode="r", encoding="utf-8").read().splitlines()
        blacklist = io.open("blacklist.txt", mode="r", encoding="utf-8").read().splitlines()

        for segment in segments:
            percent = round(segment.start/total_duration, 2)*100
            process_progressbar.configure(value=percent, mode='determinate')
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
                #if word in word_filter:
                if word not in whitelist and (word in blacklist or filter.findall(word)):
                    logging.info("Слово " + str(word) + " обнаружено на " + time.strftime('%H:%M:%S', time.gmtime(round(word_obj.start, 2))))

                    s = a[last_badword_end*1000:word_obj.start*1000]
                    audio_result = audio_result + s
                    
                    duration = (word_obj.end - word_obj.start) * 1000
                    audio_result = audio_result + a[word_obj.start*1000:word_obj.start*1000+duration*0.3]
                    s = Sine(1000).to_audio_segment(duration*0.4).apply_gain(-25)
                    audio_result = audio_result + s
                    audio_result = audio_result + a[word_obj.end*1000-duration*0.3:word_obj.end*1000]
                    last_badword_end = word_obj.end
            audio_result = audio_result + a[last_badword_end*1000:segment.end*1000]
            last_segment_end = segment.end

        # audio_result = audio_result + a[last_segment_end*1000:(last_segment_end + 2)*1000]
        # audio_result = audio_result + AudioSegment.silent(len(a[(last_segment_end + 2)*1000:]))
        audio_result = audio_result + a[last_segment_end*1000:]

        process_progressbar.configure(value=100, mode='determinate')
        style.configure('text.Horizontal.TProgressbar', text='100%')

        wavfile_export = os.path.splitext(wavfile)[0]+'_clean.wav'

        audio_result.export(wavfile_export, format='wav')

        logging.info("Готово. Результат сохранён в файл " + os.path.basename(wavfile_export))
    
    process_progressbar.configure(value=0, mode='determinate')
    process_button.configure(state=NORMAL)
    source_label.configure(state=NORMAL)
    source_button.configure(state=NORMAL)

def progressbar_load():
    global loading
    loading = True
    value = 0
    increment = 1
    while loading:
        value = value + increment
        if value == 100:
            increment = -1
        elif value == 0:
            increment = 1
        process_progressbar.configure(value=value, mode='indeterminate')
        time.sleep(0.03)
    exit()

def start_transcribe_thread():
    transcribe_thread = threading.Thread(target=transcribe)
    transcribe_thread.daemon = True
    transcribe_thread.start()

def start_progressbar_thread():
    progressbar_thread = threading.Thread(target=progressbar_load)
    progressbar_thread.daemon = True
    progressbar_thread.start()

def set_model(value):
    global model
    model = value

def censor_scale_text(value):
   censor_label.config(text = "Длительность звука цензуры (" + str(value.split(".")[0]) + "%)")

if __name__ == '__main__':
    gui = Tk()
    gui.title('Audio Censor')
    gui.geometry('900x650')
    gui.minsize(900, 650)
    gui.columnconfigure(tuple(range(60)), weight=1)

    source_frame = ttk.Frame(gui)
    source_frame.pack(side=TOP, fill=X, expand=300)
    source_label = ttk.Label(source_frame, text="Источник")
    source_label.pack(side=LEFT, padx=5, pady=5)
    source_field = ttk.Entry(source_frame, state=DISABLED)
    source_field.pack(side=LEFT, padx=5, pady=5, fill=X, expand=300)
    source_button = ttk.Button(source_frame, text='Обзор', command=select_file)
    source_button.pack(side=LEFT, padx=5, pady=5)
    process_button = ttk.Button(source_frame, text='Запуск', command=start_transcribe_thread, state=DISABLED)
    process_button.pack(side=LEFT, padx=5, pady=5)
    
    style = ttk.Style(gui)
    style.layout('text.Horizontal.TProgressbar', 
             [('Horizontal.Progressbar.trough',
               {'children': [('Horizontal.Progressbar.pbar',
                              {'side': 'left', 'sticky': 'ns'})],
                'sticky': 'nswe'}), 
              ('Horizontal.Progressbar.label', {'sticky': 'nswe'})])
    process_progressbar = ttk.Progressbar(gui, orient="horizontal", maximum=100, value=0, style='text.Horizontal.TProgressbar', mode='determinate')
    process_progressbar.pack(side=TOP, fill=X)
    process_progressbar.columnconfigure(0, weight=1)

    options_frame = ttk.Frame(gui)
    options_frame.pack(side=TOP)
    #languages = ["Afrikaans","Albanian","Amharic","Arabic","Armenian","Assamese","Azerbaijani","Bashkir","Basque","Belarusian","Bengali","Bosnian","Breton","Bulgarian","Burmese","Castilian","Catalan","Chinese","Croatian","Czech","Danish","Dutch","English","Estonian","Faroese","Finnish","Flemish","French","Galician","Georgian","German","Greek","Gujarati","Haitian","Haitian" "Creole","Hausa","Hawaiian","Hebrew","Hindi","Hungarian","Icelandic","Indonesian","Italian","Japanese","Javanese","Kannada","Kazakh","Khmer","Korean","Lao","Latin","Latvian","Letzeburgesch","Lingala","Lithuanian","Luxembourgish","Macedonian","Malagasy","Malay","Malayalam","Maltese","Maori","Marathi","Moldavian","Moldovan","Mongolian","Myanmar","Nepali","Norwegian","Nynorsk","Occitan","Panjabi","Pashto","Persian","Polish","Portuguese","Punjabi","Pushto","Romanian","Russian","Sanskrit","Serbian","Shona","Sindhi","Sinhala","Sinhalese","Slovak","Slovenian","Somali","Spanish","Sundanese","Swahili","Swedish","Tagalog","Tajik","Tamil","Tatar","Telugu","Thai","Tibetan","Turkish","Turkmen","Ukrainian","Urdu","Uzbek","Valencian","Vietnamese","Welsh","Yiddish","Yoruba"]
    languages = ["Русский"]
    language_label = ttk.Label(options_frame, text="Язык")
    language_label.configure(state=DISABLED)
    language_label.grid(row=0, column=0, padx=5, pady=5, sticky=E)
    language = StringVar(options_frame)
    language.set("Русский")
    language_optionmenu = ttk.OptionMenu(options_frame, language, *languages)
    language_optionmenu.configure(state=DISABLED)
    language_optionmenu.grid(row=0, column=1, padx=5, pady=5, sticky=W)

    #models = ["tiny.en","tiny","base.en","base","small.en","small","medium.en","medium","large-v1","large-v2","large"]
    models = ["tiny","base","small","medium","large"]
    model_label = ttk.Label(options_frame, text="Модель")
    model_label.grid(row=1, column=0, padx=5, pady=5, sticky=E)
    model = StringVar(options_frame)
    model.set("large")
    model_optionmenu = ttk.OptionMenu(options_frame, model, "large", *models)
    model_optionmenu.grid(row=1, column=1, padx=5, pady=5, sticky=W)

    theme_label = ttk.Label(options_frame, text="Тема")
    theme_label.grid(row=2, column=0, padx=5, pady=5, sticky=E)
    theme_button = ttk.Button(options_frame, text="Переключить", command=sv_ttk.toggle_theme)
    theme_button.grid(row=2, column=1, padx=5, pady=5, sticky=W)

    censor_label = ttk.Label(options_frame, text="Длительность звука цензуры (75%)")
    censor_label.grid(row=3, column=0, padx=5, pady=5, sticky=E)
    censor_scale = ttk.Scale(options_frame, from_=0, to=100, value=75, orient=HORIZONTAL, command=censor_scale_text)
    censor_scale.grid(row=3, column=1, padx=5, pady=5, sticky=W)

    log_frame = ttk.Frame(gui)
    log_frame.pack(side=TOP)
    process_log = Text(log_frame, wrap="none", width=2000, height=31)
    process_log.pack(side=LEFT)
    xsb = ttk.Scrollbar(gui, orient = HORIZONTAL)
    ysb = ttk.Scrollbar(log_frame)
    ysb.configure(command = process_log.yview)
    ysb.pack(side=RIGHT)
    ysb.lift()
    xsb.configure(command = process_log.xview)
    xsb.pack(side=TOP)
    xsb.lift()
    process_log.configure(yscrollcommand = ysb.set, xscrollcommand = xsb.set)

    # process_log = Text(gui, wrap="none", height=2000, width=2000, state=DISABLED)
    # process_log.pack()
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

    sv_ttk.use_dark_theme()

    gui.mainloop()