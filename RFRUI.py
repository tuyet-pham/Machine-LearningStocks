import tkinter as tk
from tkinter import Label, Entry
from tkinter import Frame, Button, Grid
from tkinter import TOP, LEFT, RIGHT, BOTTOM, Y, BOTH, END, N, W, E, S, DISABLED, CENTER
from tkinter import TRUE, FALSE
from tkinter import Toplevel, Message
from tkinter import PanedWindow
from tkinter import Text, Scrollbar
from tkinter import PhotoImage, Menu, colorchooser, OptionMenu, Canvas
from tkinter.filedialog import askopenfilename, Radiobutton, LabelFrame




class RButtonDark(Button):
    def __init__(self, *args, **kwargs):
        Button.__init__(self, *args, **kwargs)
        self['relief'] = 'flat'
        
        if kwargs.get('width',0) == 0:
            self['width'] = 14
        else:
            self['width'] = kwargs.get('width',0)
        
        if kwargs.get('height',0) == 0:
            self['height'] = 5
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
    

class RLabel(Label):
    def __init__(self, *args, **kwargs):
        Label.__init__(self, *args, **kwargs)        
        
        if kwargs.get('bg', '') == '':
            self['bg'] = 'gray13'
        else:
            self['bg'] = kwargs.get('bg', '')
        
        if kwargs.get('fg', '') == '':
            self['fg'] = 'gray50'
        else:
            self['fg'] = kwargs.get('fg','')

        if kwargs.get('font', '') == '':
            self['font'] = ('Helvetica', 18, "bold")
        else:
            self['font'] = kwargs.get('font','')
        
        

class REntry(Entry):
    def __init__(self, *args, **kwargs):
        Entry.__init__(self, *args, **kwargs)        
        
        self['bd'] = 0
        self['highlightthickness'] = 0
        self['relief'] ='flat'
        self['selectbackground'] = 'gray30'
        
        if kwargs.get('bg', '') == '':
            self['bg'] = 'gray20'
        else:
            self['bg'] = kwargs.get('bg', '')
        
        if kwargs.get('fg', '') == '':
            self['fg'] = 'gray40'
        else:
            self['fg'] = kwargs.get('fg','')

        if kwargs.get('font', '') == '':
            self['font'] = ('Helvetica', 11, "normal")
        else:
            self['font'] = kwargs.get('font','')
        
        if kwargs.get('width',0) == 0:
            self['width'] = 10
        else:
            self['width'] = kwargs.get('width',0)
    


class RRadiobutton(Radiobutton):
    def __init__(self, *args, **kwargs):
        Radiobutton.__init__(self, *args, **kwargs)        
        
        self['bd'] = 0
        self['highlightthickness'] = 0
        self['relief'] ='flat'
        
        if kwargs.get('bg', '') == '':
            self['bg'] = 'gray13'
            self['activebackground'] = self['bg']
        else:
            self['bg'] = kwargs.get('bg', '')
            self['activebackground'] = self['bg']
        
        self['activeforeground'] = 'MediumSlateBlue'


        if kwargs.get('fg', '') == '':
            self['fg'] = 'gray45'
        else:
            self['fg'] = kwargs.get('fg','')

        if kwargs.get('font', '') == '':
            self['font'] = ('Helvetica', 11, "normal")
        else:
            self['font'] = kwargs.get('font','')
        
    