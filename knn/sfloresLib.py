#version 1.2
##############################
#                            #
#     Made with Love by      #
#    Sergio Flores Labra     #
#                            #
##############################

import tkinter
from tkinter import ttk , messagebox, filedialog
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
 
class Ventana(ttk.Frame):
    """The adders gui and functions."""
    def __init__(self, parent, *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.root = parent
        self.root.lift()
        self.root.attributes('-topmost',True)
        self.root.after_idle(self.root.attributes,'-topmost',False)
        self.buttons = []
        center(self.root)

    def keepAlive(self):
        self.root.mainloop()

    def setSize(self,ancho,alto):
        self.root.minsize(width=ancho, height=alto)

    def setResizable(self,opcion):
        self.root.resizable(width=opcion, height=opcion)

    def setCanvas(self,ancho,alto,color):
        w = tkinter.Canvas(self.root, bg=color, width=ancho, height=alto)
        self.canvas = w
        self.canvas.pack()
        return w

    def graphPlot(self,x,y,xLabel,yLabel,title):
        f = Figure(figsize=(5,5), dpi=100)
        a = f.add_subplot(111,xlabel=xLabel,ylabel=yLabel,title=title)
        a.plot(x,y)
        self.canvas = FigureCanvasTkAgg(f,self.root)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=tkinter.BOTTOM, fill=tkinter.BOTH, expand=True)
        toolbar = NavigationToolbar2TkAgg(self.canvas, self.root)
        toolbar.update()
        self.canvas._tkcanvas.pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)

    def quitEvent(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.root.destroy()

    def setCloseConfirm(self):
        self.root.protocol("WM_DELETE_WINDOW", self.quitEvent)

    def closeWindows(self):
        quit()
 
    def setTitle(self, text):
        self.root.title(text)

    def setColor(self, color):
        self.root.configure(background=color)

    def createButton(self, text, command):
        b = tkinter.Button(self.root, text=text, command=command)
        b.pack()
        self.buttons.append({'key':text,'value':b})
        return b

    def deleteButton(self, key):
        for i in range(len(self.buttons)):
            if(self.buttons[i]['key'] == key):
                self.buttons[i]['value'].destroy()
                self.buttons.pop(i)

    def deleteAllButton(self):
        for i in range(len(self.buttons)):
            self.buttons[i]['value'].destroy()
        self.buttons.clear()

    def createLabel(self, text):
        ta = tkinter.Label(self.root, text= text)
        ta.pack()
        return ta

    def askopenfile(self,modo):
        return tkinter.filedialog.askopenfilename(filetypes=modo)


def createWindow():
    root = tkinter.Tk()
    a = Ventana(root)
    return a

def createImage(urlImagen):
    try: 
        return tkinter.PhotoImage(file = urlImagen)
    except Exception as e:
        print("we only have support for gif format extension or ")
        print(e)

def mouseEvent(event):
    print ("clicked: ", event.x, event.y)

def keyEvent(event):
    print ("Key Pressed: ", repr(event.keysym) )

def center(win):
    win.update_idletasks()
    width = win.winfo_width()
    height = win.winfo_height()
    x = (win.winfo_screenwidth() // 2) - (width // 2)
    y = (win.winfo_screenheight() // 2) - (height // 2)
    win.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    