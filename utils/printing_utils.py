#!/usr/bin/python

import colorama
from colorama import Fore, Back, Style

colorama.init()


def style(content, fore, back, t_style, form):
    string = form.format(content)
    return "{}{}{}{}{}".format(t_style, fore, back, string, Style.RESET_ALL)


def show(content, alternate=False):
    return style(content, Fore.CYAN if alternate else Fore.GREEN, Back.RESET, Style.BRIGHT, "{}")


def err(content):
    return style(content, Fore.RED, Back.RESET, Style.NORMAL, "{}")


def fatal(content):
    return style(content, Fore.RED, Back.RESET, Style.BRIGHT, "{}")


def warn(content):
    return style(content, Fore.YELLOW, Back.RESET, Style.NORMAL, "{}")


def accent(content):
    return style(content, Fore.GREEN, Back.RESET, Style.NORMAL, "{}")


def param(content):
    return style(content, Fore.BLUE, Back.RESET, Style.NORMAL, "{}")


def show_int(content, alternate=False):
    return style(content, Fore.CYAN if alternate else Fore.GREEN, Back.RESET, Style.BRIGHT, "{:d}")


def accent_int(content):
    return style(content, Fore.GREEN, Back.RESET, Style.NORMAL, "{:d}")


def param_int(content):
    return style(content, Fore.BLUE, Back.RESET, Style.NORMAL, "{:d}")


def show_e(content, alternate=False):
    return style(content, Fore.CYAN if alternate else Fore.GREEN, Back.RESET, Style.BRIGHT, "{:.3e}")


def accent_e(content):
    return style(content, Fore.GREEN, Back.RESET, Style.NORMAL, "{:.3e}")


def param_e(content):
    return style(content, Fore.BLUE, Back.RESET, Style.NORMAL, "{:.3e}")


def show_f(content, alternate=False):
    return style(content, Fore.CYAN if alternate else Fore.GREEN, Back.RESET, Style.BRIGHT, "{:.3f}")


def accent_f(content):
    return style(content, Fore.GREEN, Back.RESET, Style.NORMAL, "{:.3f}")


def param_f(content):
    return style(content, Fore.BLUE, Back.RESET, Style.NORMAL, "{:.3f}")

