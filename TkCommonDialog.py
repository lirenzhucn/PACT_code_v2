#!/usr/bin/env python

from Tkinter import Tk, StringVar, IntVar  # helper classes
from Tkinter import END, LEFT, ACTIVE  # constants
from Tkinter import Entry, Frame, Button, Checkbutton  # widget


# a check button widget that has a getVal method
class ValCheckbutton(Checkbutton):

    def __init__(self, master, **kw):
        self.intVal = IntVar()
        Checkbutton.__init__(self, master, var=self.intVal, **kw)

    def setVal(self, val):
        if bool(val):
            self.select()
        else:
            self.deselect()

    def getVal(self):
        return bool(self.intVal.get())


class NumberEntry(Entry):

    def __init__(self, master, **kw):
        self.strVar = StringVar()
        vcmd = (master.register(self.isValid),
                '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
        Entry.__init__(self, master, textvariable=self.strVar,
                       validate='key', validatecommand=vcmd, **kw)

    def setVal(self, val):
        self.delete(0, END)
        self.insert(0, '%.2f' % (val))

    def getVal(self):
        return float(self.strVar.get())

    def isValid(self, action, index, value_if_allowed, prior_value,
                text, validation_type, trigger_type, widget_name):
        try:
            float(value_if_allowed)
            return True
        except ValueError:
            return False


class CommonDialog(Tk):

    def __init__(self, title=None):
        Tk.__init__(self)
        if title:
            self.title(title)
        # check if 'result' variable has been defined by the user yet
        if 'result' not in self.__dict__:
            self.result = None
        body = Frame(self)
        self.initial_focus = self.body(body)
        body.pack(padx=5, pady=5)
        self.buttonbox()
        self.grab_set()
        if not self.initial_focus:
            self.initial_focus = self
        self.protocol('WM_DELETE_WINDOW', self.cancel)
        self.initial_focus.focus_set()
        self.wait_window(self)

    # construction hooks
    def body(self, master):
        # Create dialog body, return widget that should have initial
        # focus. This method should be overridden
        return None

    def buttonbox(self):
        # add standard button box.
        box = Frame(self)
        w = Button(box, text='OK', width=10, command=self.ok, default=ACTIVE)
        w.pack(side=LEFT, padx=5, pady=5)
        w = Button(box, text='Cancel', width=10, command=self.cancel)
        w.pack(side=LEFT, padx=5, pady=5)
        self.bind('<Return>', self.ok)
        self.bind('<Escape>', self.cancel)
        box.pack()

    # standard button semantics
    def ok(self, event=None):
        if not self.validate():
            self.initial_focus.focus_set()
            return
        self.withdraw()
        self.update_idletasks()
        self.apply()
        self.cancel()

    def cancel(self, event=None):
        self.destroy()

    # command hooks
    def validate(self):
        return True

    def apply(self):
        pass
