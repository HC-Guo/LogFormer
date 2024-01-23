import Drain

input_dir = 'log_data/'  # The input directory of log file
output_dir = 'parse_result/'  # The output directory of parsing results
log_file = 'openstack.log'  # The input log file name

hdfs_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'  # HDFS log format
bgl_format = '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>'
openstack_format = '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>'

bgl_regex = [
    r'core\.\d+',
    r'(?:\/[\*\w\.-]+)+',  # path
    r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)',  # IP
    r'0x[0-9a-f]+(?: [0-9a-f]{8})*',  # hex
    r'[0-9a-f]{8}(?: [0-9a-f]{8})*',
    r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',  # Numbers
]

hdfs_regex = [
    r'blk_(|-)[0-9]+',  # block id
    r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)',  # IP
    r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',  # Numbers
]

openstack_regex = [
    r'(?<=\[instance: ).*?(?=\])',
    r'(?<=\[req).*(?= -)',
    r'(?<=image ).*(?= at)',
    r'(?<=[^a-zA-Z0-9])(?:\/[\*\w\.-]+)+',  # path
    r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)',  # IP
    r'(?<=\s|=)\d+(?:\.\d+)?'
]


st = 0.5  # Similarity threshold
depth = 4  # Depth of all leaf nodes
# depth = 5  # openstack

parser = Drain.LogParser(openstack_format, indir=input_dir,
                         outdir=output_dir, depth=5, st=0.5, rex=openstack_regex)
parser.parse(log_file)
