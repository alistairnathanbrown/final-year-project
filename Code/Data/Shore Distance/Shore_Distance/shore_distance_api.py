import numpy
import vptree
from geographiclib.geodesic import Geodesic

def geoddist(p1, p2):
  # p1 = [lon1, lat1] in degrees
  # p2 = [lon2, lat2] in degrees
  return Geodesic.WGS84.Inverse(p1[1], p1[0], p2[1], p2[0])['s12']

coast = vptree.VPTree(numpy.loadtxt('coast.txt'), geoddist)
print('vessel closest-coast dist')
for v in numpy.loadtxt('vessels.txt'):
  c = coast.get_nearest_neighbor(v)
  print(list(v), list(c[1]), c[0])