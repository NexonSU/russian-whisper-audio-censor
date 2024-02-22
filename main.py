import glob
import io
import logging
import os
import sys
import threading
import time
import faster_whisper
import torch
import psutil
import re
from pydub import AudioSegment
from pydub.generators import Sine
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
import sv_ttk

logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='[%(levelname)s] %(asctime)s: %(message)s')

model = ""
wavfiles = []
start_time = 0
loading = False

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
                logging.info("Обнаружен файл: " + file)
    else:
        messagebox.showwarning(title="Файлы не найдены", message="В указанной папке нет wav-файлов.")

def whisper_transcribe():
    global loading

    set_gui_state(DISABLED)

    with open(os.path.dirname(wavfiles[0]) + "/stats.txt", "a", encoding="utf8") as stats:
        stats.write("--- " + time.strftime('%Y.%m.%d %H:%M:%S') + "\n")

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

        modelname = faster_whisper.WhisperModel(model.get().removeprefix("whisper-"))
        segments, info = modelname.transcribe(audio=wavfile, word_timestamps=True, condition_on_previous_text=False, vad_filter=True, no_speech_threshold=0.5)
        total_duration = round(info.duration, 2)

        loading = False
        process_progressbar.configure(value=0, mode='determinate')
        style.configure('text.Horizontal.TProgressbar', text='0%', anchor='center')
        start_time = time.time()

        a = AudioSegment.from_wav(wavfile)
        last_end = 0
        badword_count = 0
        transcription = ""
        scale_badword = censor_scale.get()/100
        scale_audio = (1-scale_badword)/2
        regex = re.compile('[\W+]')
        filter = re.compile('(?<![а-яё])(?:(?:(?:у|[нз]а|(?:хитро|не)?вз?[ыьъ]|с[ьъ]|(?:и|ра)[зс]ъ?|(?:о[тб]|п[оа]д)[ьъ]?|(?:\S(?=[а-яё]))+?[оаеи-])-?)?(?:[её](?:б(?!о[рй]|рач)|п[уа](?:ц|тс))|и[пб][ае][тцд][ьъ]).*?|(?:(?:н[иеа]|ра[зс]|[зд]?[ао](?:т|дн[оа])?|с(?:м[еи])?|а[пб]ч)-?)?ху(?:[яйиеёю]|л+и(?!ган)).*?|бл(?:[эя]|еа?)(?:[дт][ьъ]?)?|\S*?(?:п(?:[иеё]зд|ид[аое]?р|ед(?:р(?!о)|[аое]р|ик))|бля(?:[дбц]|тс)|[ое]ху[яйиеёю]|хуйн).*?|(?:о[тб]?|про|на|вы)?м(?:анд(?:[ауеыи](?:л(?:и[сзщ])?[ауеиы])?|ой|[ао]в.*?|юк(?:ов|[ауи])?|е[нт]ь|ища)|уд(?:[яаиое].+?|е?н(?:[ьюия]|ей))|[ао]л[ао]ф[ьъ](?:[яиюе]|[еёо]й))|елд[ауые].*?|ля[тд]ь|(?:[нз]а|по)х)(?![а-яё])')
        whitelist = io.open("whitelist.txt", mode="r", encoding="utf-8").read().splitlines()
        blacklist = io.open("blacklist.txt", mode="r", encoding="utf-8").read().splitlines()

        for segment in segments:
            percent = round(segment.start/total_duration, 2)*100
            process_progressbar.configure(value=percent, mode='determinate')
            if percent > 0 and percent < 99:
                eta = round(((time.time() - start_time)/percent * (100 - percent)))
                style.configure('text.Horizontal.TProgressbar', text=str(round(percent))+'% ETA: ' + time.strftime('%H:%M:%S', time.gmtime(eta)))
            if percent == 100:
                style.configure('text.Horizontal.TProgressbar', text='99%')
            current_line = time.strftime('%H:%M:%S: ', time.gmtime(round(segment.start, 2))) + segment.text.removeprefix(" ")
            logging.info(current_line)
            transcription += current_line + "\n"
            for word_obj in segment.words:
                word = regex.sub('', word_obj.word).lower()
                logging.debug("Проверяем " + word + " на " + time.strftime('%H:%M:%S', time.gmtime(round(word_obj.start, 2))) + " с " + str(word_obj.probability))
                if word not in whitelist and (word in blacklist or filter.findall(word)):
                    logging.info("Слово " + str(word) + " обнаружено на " + time.strftime('%H:%M:%S', time.gmtime(round(word_obj.start, 2))))
                    badword_count += 1
                    audio_result += a[last_end*1000:word_obj.start*1000]
                    duration = (word_obj.end - word_obj.start) * 1000
                    audio_result += a[word_obj.start*1000:word_obj.start*1000+duration*scale_audio]
                    audio_result += Sine(1000).to_audio_segment(duration*scale_badword).apply_gain(-25)
                    audio_result += a[word_obj.end*1000-duration*scale_audio:word_obj.end*1000]
                    last_end = word_obj.end

        audio_result += a[last_end*1000:]

        process_progressbar.configure(value=100, mode='determinate')
        style.configure('text.Horizontal.TProgressbar', text='100%')

        wavfile_export = os.path.splitext(wavfile)[0]+'_clean.wav'
        transcription_export = os.path.splitext(wavfile)[0]+'_transcription.txt'

        with open(transcription_export, "w", encoding="utf8") as text_file:
            text_file.write(transcription)
        audio_result.export(wavfile_export, format='wav')

        with open(os.path.dirname(wavfile) + "/stats.txt", "a", encoding="utf8") as stats:
            stats.write(os.path.basename(wavfile) + " содержит матов: " + str(badword_count) + ".\n")

        logging.info("Матов обнаружено: " + str(badword_count) + ".")
        logging.info("Результат сохранён в файл " + os.path.basename(wavfile_export))
    
    logging.info("Все операции завершены.")
    process_progressbar.configure(value=0, mode='determinate')
    set_gui_state(NORMAL)

def set_gui_state(state):
    process_button.configure(state=state)
    source_label.configure(state=state)
    source_button.configure(state=state)
    model_label.configure(state=state)
    model_optionmenu.configure(state=state)
    censor_label.configure(state=state)
    censor_scale.configure(state=state)

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
    sys.exit(0)

def start_transcribe_thread():
    if "whisper" in model.get():
        transcribe_thread = threading.Thread(target=whisper_transcribe)
    transcribe_thread.daemon = True
    transcribe_thread.start()

def start_progressbar_thread():
    progressbar_thread = threading.Thread(target=progressbar_load)
    progressbar_thread.daemon = True
    progressbar_thread.start()

def set_debug_level(value):
    global logger
    if value == "CRITICAL":
        logging.getLogger().setLevel(logging.CRITICAL)
    elif value == "ERROR":
        logging.getLogger().setLevel(logging.ERROR)
    elif value == "WARN":
        logging.getLogger().setLevel(logging.WARN)
    elif value == "INFO":
        logging.getLogger().setLevel(logging.INFO)
    elif value == "DEBUG":
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.NOTSET)

def censor_scale_text(value):
   censor_label.config(text = "Длительность звука цензуры (" + str(value.split(".")[0]) + "%)")

if __name__ == '__main__':
    if not os.path.isfile("whitelist.txt"):
        with open("whitelist.txt", "w", encoding="utf8") as file:
            file.write("мудак")

    if not os.path.isfile("blacklist.txt"):
        with open("blacklist.txt", "w", encoding="utf8") as file:
            file.write("хуй")
    
    gui = Tk()
    gui.title('Audio Censor')
    gui.geometry('600x150')
    gui.resizable(False, False)
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
    languages = ["Авто"]
    language_label = ttk.Label(options_frame, text="Язык")
    language_label.configure(state=DISABLED)
    language_label.grid(row=0, column=0, padx=5, pady=5, sticky=E)
    language = StringVar(options_frame)
    language.set("Авто")
    language_optionmenu = ttk.OptionMenu(options_frame, language, *languages)
    language_optionmenu.configure(state=DISABLED)
    language_optionmenu.grid(row=0, column=1, padx=5, pady=5, sticky=W)

    #models = ["tiny.en","tiny","base.en","base","small.en","small","medium.en","medium","large-v1","large-v2","large"]
    models = ["whisper-tiny","whisper-base","whisper-small","whisper-medium","whisper-large"]
    model_label = ttk.Label(options_frame, text="Модель")
    model_label.grid(row=0, column=2, padx=5, pady=5, sticky=E)
    model = StringVar(options_frame)
    model.set("whisper-large")
    model_optionmenu = ttk.OptionMenu(options_frame, model, "whisper-large", *models)
    model_optionmenu.grid(row=0, column=3, padx=5, pady=5, sticky=W)

    debug_levels = ["NOTSET","DEBUG","INFO","WARN","ERROR","CRITICAL"]
    debug_label = ttk.Label(options_frame, text="Уровень лога")
    debug_label.grid(row=0, column=4, padx=5, pady=5, sticky=E)
    debug_level = StringVar(options_frame)
    debug_level.set("INFO")
    debug_optionmenu = ttk.OptionMenu(options_frame, debug_level, "INFO", *debug_levels, command=set_debug_level)
    debug_optionmenu.grid(row=0, column=5, padx=5, pady=5, sticky=W)

    options_frame2 = ttk.Frame(gui)
    options_frame2.pack(side=TOP)
    theme_label = ttk.Label(options_frame2, text="Тема")
    theme_label.grid(row=0, column=0, padx=5, pady=5, sticky=E)
    theme_button = ttk.Button(options_frame2, text="Переключить", command=sv_ttk.toggle_theme)
    theme_button.grid(row=0, column=1, padx=5, pady=5, sticky=W)
    censor_label = ttk.Label(options_frame2, text="Длительность звука цензуры (75%)")
    censor_label.grid(row=0, column=2, padx=5, pady=5, sticky=E)
    censor_scale = ttk.Scale(options_frame2, from_=0, to=100, value=75, orient=HORIZONTAL, command=censor_scale_text)
    censor_scale.grid(row=0, column=3, padx=5, pady=5, sticky=W)

    logging.info("Укажите папку с wav файлами.")

    sv_ttk.use_dark_theme()

    gui.mainloop()