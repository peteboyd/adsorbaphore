
from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Integer, Float, String, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

atom_associations = Table('atom_associations',
                          Base.metadata,
                          Column('active_site_name', Text,
                                 ForeignKey('active_sites.name')),
                          Column('atom_name', Text,
                                 ForeignKey('active_site_atoms.name')))

distance_associations = Table('distance_associations',
                              Base.metadata,
                              Column('active_site_name', Text,
                                     ForeignKey('active_sites.name')),
                              Column('distance_matrix_name', Text,
                                     ForeignKey('distance_matrix.name')))
class SQL_Pharma(Base):
    """SQLAlchemy model for the resulting pharmacophores"""
    __tablename__ = 'pharmacophores'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    rank = Column(Integer)
    length = Column(Integer)
    c_prob = Column(Text)
    o_prob = Column(Text)
    #vdw_dist = Column(Text)
    #el_dist = Column(Text)
    e_dist = Column(Text)
    vdwe_av = Column(Float)
    vdwe_std = Column(Float)
    ele_av = Column(Float)
    ele_std = Column(Float)
    error = Column(Float)

    def __init__(self, name, vdwe_av, vdwe_std, ele_av, ele_std, length):
        self.name = name
        self.vdwe_av = vdwe_av
        self.vdwe_std = vdwe_std
        self.ele_av = ele_av
        self.ele_std = ele_std
        self.length = length

    def set_rank(self, rank):
        self.rank = rank

    def set_error(self, error):
        self.error = error

    def set_probs(self, c_prob, o_prob, e_dist):
        self.c_prob = c_prob
        self.o_prob = o_prob
        #self.vdw_dist = vdw_dist
        #self.el_dist = el_dist
        self.e_dist = e_dist


class SQL_ActiveSiteAtoms(Base):
    __tablename__ = "active_site_atoms"
    id = Column(Integer, primary_key=True)
    x = Column(Float)
    y = Column(Float)
    z = Column(Float)
    elem = Column(Text)
    charge = Column(Float)
    orig_id = Column(Integer)
    mof_id = Column(Integer)
    name = Column(Text, ForeignKey('active_sites.name'))

    def __init__(self, pos, elem, charge, oid, mofid, name):
        self.x = pos[0]
        self.y = pos[1]
        self.z = pos[2]
        self.elem = elem
        self.charge = charge
        self.orig_id = oid
        self.mof_id = mofid
        self.name = name

class SQL_ActiveSite(Base):
    __tablename__ = "active_sites"
    id = Column(Integer, primary_key=True)
    size = Column(Integer)
    vdweng = Column(Float)
    eleng = Column(Float)
    mofpath = Column(Text)
    name = Column(Text)
    atoms = relationship('SQL_ActiveSiteAtoms',backref='active_sites')
    distances = relationship('SQL_Distances',backref='active_sites')
    co2 = relationship('SQL_ActiveSiteCO2', backref='carbon_dioxide')

    def __init__(self, name, size, mofpath, vdweng, eleng):
        self.name = name
        self.size = size
        self.vdweng = vdweng
        self.eleng = eleng
        self.mofpath = mofpath

class SQL_Distances(Base):
    __tablename__ = "distance_matrix"
    id = Column(Integer, primary_key=True)
    row = Column(Integer)
    col = Column(Integer)
    dist = Column(Float)
    name = Column(Text, ForeignKey('active_sites.name'))

    def __init__(self, row, column, dist, name):
        self.row = row
        self.col = column
        self.dist = dist
        self.name = name

class SQL_ActiveSiteCO2(Base):
    __tablename__ = "carbon_dioxide"
    id = Column(Integer, primary_key=True)
    name = Column(Text, ForeignKey('active_sites.name'))
    cx = Column(Float)
    cy = Column(Float)
    cz = Column(Float)
    o1x = Column(Float)
    o1y = Column(Float)
    o1z = Column(Float)
    o2x = Column(Float)
    o2y = Column(Float)
    o2z = Column(Float)
    
    def __init__(self, name, pos):
        self.name = name
        self.cx = pos[0][0]
        self.cy = pos[0][1]
        self.cz = pos[0][2]
        self.o1x = pos[1][0]
        self.o1y = pos[1][1]
        self.o1z = pos[1][2]
        self.o2x = pos[2][0]
        self.o2y = pos[2][1]
        self.o2z = pos[2][2]

class SQL_Adsorbophore(Base):
    __tablename__ = "adsorbophore"
    rank = Column(Integer, primary_key=True)
    active_sites = relationship('SQL_AdsorbophoreSite',backref='adsorbophore')
    def __init__(self, rank):
        self.rank = rank

class SQL_AdsorbophoreSite(Base):
    __tablename__ = "adsorbophore_site"
    id = Column(Integer, primary_key=True)
    rank = Column(Integer, ForeignKey('adsorbophore.rank'))
    name = Column(Text)
    indices = relationship('SQL_AdsorbophoreSiteIndices',backref='adsorbophore_site')
    def __init__(self, rank, name):
        self.rank = rank
        self.name = name

class SQL_AdsorbophoreSiteIndices(Base):
    __tablename__ = "adsorbophore_site_indices"
    id = Column(Integer, primary_key=True)
    index = Column(Integer)
    name = Column(Text, ForeignKey('adsorbophore_site.name'))
    def __init__(self, name, index):
        self.index = index
        self.name = name

class Data_Storage(object):
    """Container for each pharmacophore. Contains all properties calculated for each pharma"""

    def __init__(self, db_name):
        self.engine = create_engine('sqlite:///%s.db'%(db_name))
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def store(self, sql_pharma):
        self.session.add(sql_pharma)

    def flush(self):
        self.session.flush()
        self.session.commit()

    def get_active_site(self, name):
        return self.session.query(SQL_ActiveSite).filter(SQL_ActiveSite.name == name).first()
