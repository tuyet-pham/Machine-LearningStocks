import tkinter as tk
from tkinter import Label, Entry
from tkinter import Frame, Button, Grid
from tkinter import TOP, LEFT, RIGHT, BOTTOM, Y, BOTH, END, N, W, E, S, DISABLED
from tkinter import TRUE, FALSE
from tkinter import Toplevel, Message
from tkinter import PanedWindow
from tkinter import Text, Scrollbar
from tkinter import PhotoImage, Menu, colorchooser, OptionMenu
from tkinter.filedialog import askopenfilename




class RButtonDark(Button):
    def __init__(self, *args, **kwargs):
        Button.__init__(self, *args, **kwargs)
        self['relief'] = 'flat'
        self['width'] = 10
        if kwargs.get('height',0) == 0:
            self['height'] = 4
        else:
            self['height'] = kwargs.get('height',0)
        self['bd'] = 0
        self['highlightthickness'] = 0
        self['activebackground'] = 'gray30'
        self['activeforeground'] = 'MediumSlateBlue'
        
        if kwargs.get('bg', '') == '':
            self['bg'] = 'gray27'
        else:
            self['bg'] = kwargs.get('bg', '')
            
        self['font'] = ('Helvetica Neue', 9)
        
        if kwargs.get('fg', '') == '':
            self['fg'] = 'gray65'
        else:
            self['fg'] = kwargs.get('fg','')


class RMenu(Menu):
    def __init__(self, *args, **kwargs):
        Menu.__init__(self, *args, **kwargs)
        self['bg'] = 'MediumSlateBlue'
        self['fg'] = 'gray80'
        self['font'] = ('Helvetica Neue', 9)
        self['relief'] = 'flat'
        self['activebackground'] = 'SlateBlue'
        self['activeborderwidth'] = 0
        self['borderwidth'] = 0
    
    