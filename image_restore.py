import os,sys

class Image:
    def __init__(self,filename):
        self.filename=filename
        self.image=open(filename)
        return

    def open_file(filename):
        return open(filename,"rb")

    def restore(self):
        print ("restore!")

    def save_as(filename):
        pass

if __name__ == "__main__":
    for i in sys.argv[1:]:
        print(i)
