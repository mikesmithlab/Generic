'''
Some simple helper functions for manipulating files
'''



'''
save and load dictionary from file.
'''
def save_dict_to_file(filename, dic):
    f = open(filename,'w')
    f.write(str(dic))
    f.close()

def load_dict_from_file(filename):
    f = open(filename, 'r')
    data = f.read()
    f.close()
    return eval(data)