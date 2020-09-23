from collections import namedtuple
from itertools import chain, combinations


def get_ids_from_parse(fn):
    with open(fn, "r") as f:
        next(f)
        parse = next(f).strip()
        return [
            int(n.strip().split('_')[1]) for n in parse.strip("[]").split(",")]


def set_parse(fn, graph):
    with open(fn, "w+") as f:
        f.write("# IRTG unannotated corpus file, v1.0\n")
        f.write(
            "# interpretation ud: de.up.ling.irtg.algebra.graph.GraphAlgebra"
            "\n")
        f.write(graph + "\n")
        f.write("(dummy_0 / dummy_0)\n")


def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))


Token = namedtuple('Token', [
    'id', 'lemma', 'word', 'pos', 'tpos', 'misc', 'head', 'deprel',
    'comp_edge', 'space_after', 'word_id'])


def gen_tsv_sens(stream):
    curr_sen = []
    for raw_line in stream:
        line = raw_line.strip()
        if line.startswith("#"):
            continue
        if not line:
            yield curr_sen
            curr_sen = []
            continue
        curr_sen.append(line.split('\t'))


def get_conll_sen(sen):
    tokens = []
    for fields in sen:
        assert len(fields) == 10
        fields[0] = int(fields[0])
        fields[6] = int(fields[6])
        tokens.append(Token(*fields))


def get_conll_sens_from_file(fn):
    with open(fn, "r") as f:
        for sen in gen_tsv_sens(f):
            yield get_conll_sen(sen)
