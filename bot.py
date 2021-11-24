#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This program is dedicated to the public domain under the CC0 license.

"""
Simple Bot to reply to Telegram messages.

First, a few handler functions are defined. Then, those functions are passed to
the Dispatcher and registered at their respective places.
Then, the bot is started and runs until we press Ctrl-C on the command line.

Usage:
Basic Echobot example, repeats messages.
Press Ctrl-C on the command line or send a signal to the process to stop the
bot.
"""

import logging
import pickle

import pandas as pd
import numpy as np

import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


# Define a few command handlers. These usually take the two arguments update and
# context. Error handlers also receive the raised TelegramError object in error.
def start(update, context):
    """Send a message when the command /start is issued."""
    update.message.reply_text('Hi!')


def help(update, context):
    """Send a message when the command /help is issued."""
    update.message.reply_text('Help!')


def echo(update, context):
    """Echo the user message."""
    update.message.reply_text(update.message.text)


def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)


def get_predict(user_input: 'str', context):
    if user_input.message.text == '--help':
        user_input.message.reply_text('Привет! Я бот, который может определять, к какому классу относится Ваш вопрос.\n'
                                      'Для классификации я использу модель логистической регрессии.\n'
                                      'Сейчас я умею предсказывать десять классов: карты, адреса, справки, кредиты,'
                                      ' страхование, сообщения, подписки/услуги, списание средств,'
                                      ' кэшбек и бонусы, штрафы/просрочка')
    elif any(user_input.message.text in s for s in['Привет', 'Здравствуйте', 'Здраствуйте', 'Добрый день', 'Доброе утро',
                                       'Добрый вечер', 'Здраствуйте']):
        user_input.message.reply_text('Привет!')
    elif user_input.message.text == 'Спасибо':
        user_input.message.reply_text('Рад помочь!')
    else:
        print(load_clf)
        pred = user_input.message.text.translate(str.maketrans('', '', string.punctuation))
        print(1, pred)
        pred = ' '.join(tokenizer.tokenize(pred.lower()))
        print(2, pred)
        pred = load_tfidf.transform([pred])
        print(3, pred)
        pred = load_clf.predict(pred)
        print(4, pred)
        pred = label_explain[pred[0]]
        print(5, pred)
        user_input.message.reply_text(pred)


load_tfidf = pickle.load(open('tfidf.pkl', 'rb'))
tokenizer = nltk.tokenize.WordPunctTokenizer()
load_clf = pickle.load(open('bot_classifier.pkl', 'rb'))
label_explain = pd.read_csv('label_explain.csv')
label_explain = dict(label_explain.to_dict(orient='split')['data'])


def main():
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    # Make sure to set use_context=True to use the new context based callbacks
    # Post version 12 this will no longer be necessary
    updater = Updater("2122856411:AAGz4tXFjT1QsomY6MDMEleWyz41nvAZ2JU", use_context=True)


    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # on different commands - answer in Telegram
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))
    # dp.add_handler(CommandHandler("--help", for_gosha))

    # on noncommand i.e message - echo the message on Telegram
    dp.add_handler(MessageHandler(Filters.text, get_predict))

    # log all errors
    dp.add_error_handler(error)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()
