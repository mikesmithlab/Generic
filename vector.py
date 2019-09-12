import numpy as np


class Vector3:
    def __init__(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z

    def _np_vector(self):
        return np.array([self.x, self.y, self.z])

    def mag(self):
        return np.linalg.norm(np.array([self.x,self.y,self.z]))

    def unit(self):
        magnitude = self.mag()
        return Vector3(self.x/magnitude, self.y/magnitude, self.z/magnitude)

    def dot(self, other_vector):
        return np.dot(self._np_vector(), other_vector._np_vector())

    def cross(self, other_vector):
        return np.cross(self._np_vector(), other_vector._np_vector())

    def angle(self, other_vector):
        theta=np.arccos(self.dot(other_vector)/(self.mag()*np.linalg.norm(other_vector._np_vector())))*180/np.pi
        return theta

    def component(self, other_vector):
        #This is the magnitude of the projection of vector self onto other_vector
        component = self.mag()*np.cos((np.pi/180)*self.angle(other_vector))
        return component




if __name__ == '__main__':
    a = Vector3(3,0,0)
    b = Vector3(0,2,0)

    print(a.x)
    print(a.mag())
    print(a.unit())
    print(a.dot(b))
    print(a.dot(a))
    print(a.cross(b))
    print(a.cross(a))
    print(a.angle(b))