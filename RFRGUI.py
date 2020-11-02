
import pandas as pd
from sklearn import tree
import numpy as np
import matplotlib as plot
from RFRUI import *
# from RFR_stocks import pick_stock

# tk.Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
# filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
# print(filename)

    
# ----------- Important switch class for ALL frames ----------- #
class Switch(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.setdefault()
        self._frame = None
        

        self.switch_frame(EntryMenu)
        
    

    def switch_frame(self, frame_class):
        # if(frame_class == EntryMenu):
        #     self.buttonnav.configure(state=tk.DISABLED)
        # else:
        #     self.buttonnav.configure(state=tk.ENABLED)
            
        new_frame = frame_class(self)
        if self._frame is not None:
            self._frame.destroy()
        self._frame = new_frame
        self._frame.pack(side=RIGHT, fill=BOTH, expand=TRUE)
 
    
    
    def setdefault(self):
        self.title("Stock Prediction")
        self.geometry("700x500")
        self['bg'] = 'gray13'
        self.buttonnav = ButtonBar(self)
        self.buttonnav.pack(side=LEFT, fill=BOTH)

    
    
    def reset(self):
        frame = Toplevel(self, width=100, height=50)
        frame['bg'] = "gray"
        tk.Label(frame, text="Reset",fg='gray50', bg='gray13', font=('Helvetica', 18, "bold")).pack(side=TOP, fill="x", pady=10, padx=20)
        frame.grab_set()
        
        
        
class ButtonBar(Frame):
    def __init__(self, master, *args, **kwargs):
        Frame.__init__(self, master, *args, **kwargs)
        self.rowat = 0
        self['bg'] = 'gray27'
        
        Button_frames = {"Choose Sets": StockPicker, "View Tables": Tables, "Adjustments": Adjustments, "Create Tree PNG": MakeTree, "Intro": MainMenu}
        for k, v in Button_frames.items():
            self.add_buttons(1, master, k, v)
            
        self.add_buttons(2, master, test_lable="Quit", frame_call=quit)

        
    
    def add_buttons(self, typebutton, master, test_lable, frame_call):
        if typebutton == 1:
            RButtonDark(self, text=test_lable, command=lambda: master.switch_frame(frame_call)).grid(row=self.rowat, sticky=W+N+S)
            self.rowat = self.rowat+1
        elif typebutton == 2:
            RButtonDark(self, text=test_lable, bg='MediumSlateBlue', fg='snow', height=3, command=frame_call).grid(row=self.rowat, sticky=W+N+S)
            self.rowat = self.rowat+1
        



# ----------- Entry Page of App ----------- #
class EntryMenu(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self['bg'] = 'gray13'
        tk.Label(self, text="Stock Predictor V1.0", fg='gray50', bg='gray13', font=('Helvetica', 18, "bold")).pack(side=TOP, fill="x", pady=40)
        RButtonDark(self, text="Enter", bg='MediumSlateBlue', fg='snow', height=4, command=lambda: master.switch_frame(MainMenu)).pack(side=TOP, fill="x", pady=40)



# ----------- Main Menu Apps ----------- #
class MainMenu(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self['bg'] = 'gray13'
        tk.Label(self, text="Stock Predictor V1.0!",fg='gray50', bg='gray13', font=('Helvetica', 18, "bold")).pack(side=TOP, fill="x", pady=40)
        
        

# ----------- About App ----------- #
class StockPicker(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self['bg'] = 'gray13'
        tk.Label(self, text="Stock Picker", bg='gray13', fg='gray50', font=('Helvetica', 18, "bold")).pack(side=TOP, fill="x", pady=40)
        


# ----------- About App ----------- #
class About(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self['bg'] = 'gray13'

        self.StockM = Text(self, height=15, width=53, relief='flat')
        self.StockM.config(state='disabled')
        self.StockM.pack(side=TOP, fill=Y, pady=20)
        
        



# ----------- About App ----------- #
class Adjustments(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self['bg'] = 'gray13'
        tk.Label(self, fg='gray50',text="Adjustments", bg='gray13', font=('Helvetica', 18, "bold")).pack(side=TOP, fill="x", pady=40)
        
        


# ----------- About App ----------- #
class MakeTree(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self['bg'] = 'gray13'
        tk.Label(self, text="Make Tree PNG", fg='gray50', bg='gray13', font=('Helvetica', 18, "bold")).pack(side=TOP, fill="x", pady=40)



# ----------- About App ----------- #
class Tables(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self['bg'] = 'gray13'
        tk.Label(self, text="Tables", bg='gray13', fg='gray50', font=('Helvetica', 18, "bold")).pack(side=TOP, fill="x", pady=40)





if __name__ == "__main__":
    app = Switch()
    app.mainloop()