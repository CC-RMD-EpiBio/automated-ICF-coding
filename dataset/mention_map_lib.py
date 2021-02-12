def load(f):
    _map = {}
    with open(f, 'r') as stream:
        for line in stream:
            (m_ID, doc_ID) = [s.strip() for s in line.split('\t')]
            _map[int(m_ID)] = doc_ID
    return _map

def write(_map, f):
    with open(f, 'w') as stream:
        for (m_ID, doc_ID) in _map.items():
            stream.write('%d\t%s\n' % (m_ID, doc_ID))
