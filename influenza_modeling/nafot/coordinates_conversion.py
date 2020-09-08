from pyproj import Proj, transform


# Receives longitude & latitude and returns x,y in ITM (Israel Transverse Mercator)
def wgs_to_itm(longitude, latitude):
    prj_wgs = Proj(init='epsg:4326')
    prj_itm = Proj(init='epsg:2039')
    x, y = transform(prj_wgs, prj_itm, longitude, latitude)
    return x, y


# Receives x,y in ITM (Israel Transverse Mercator) and returns longitude & latitude
def itm_to_wgs(x,y):
    prj_wgs = Proj(init='epsg:4326')
    prj_itm = Proj(init='epsg:2039')
    longitude, latitude = transform(prj_itm, prj_wgs, x, y)
    return longitude, latitude

