import codecs
import sys

from concrete.inspect import *
from concrete.nitf import *
from concrete.util.comm_container import ZipFileBackedCommunicationContainer

# This is a list of all communications in CAW and their tgz file names
docs_list_file = open(
    "/export/a05/mahsay/domain/caw-data-list/cawiki-en.all.list", "r")
searchlines = docs_list_file.readlines()
docs_list_file.close()

with open(sys.argv[1]) as f:  # list of doc_ids
    fw = open(sys.argv[2], 'w')
    for line in f:

        # Find the .tgz file of the doc_id
        doc_id = line.rstrip('\n')
        fw.write("doc_id: " + doc_id + "\n")
        zipfile = ""
        print("doc_id: ", doc_id)
        for i, line in enumerate(searchlines):
            if "\t" + doc_id + ".comm" in line:
                tgzfile = line[0:line.find("\t")]
                zipfile = tgzfile.split(".")[0] + "." + tgzfile.split(".")[1]
                fw.write("tgz_file: " + tgzfile + "\n")
                break

        # Read the "passage" section of the communication of the doc_id
        # For speed-up tgz files are converted to zip files.
        # Currenlty fast random access to a comm using the Concrete Python API only works for zip files.
        container = ZipFileBackedCommunicationContainer(
            '/export/a05/mahsay/domain/Concrete-Lucene/caw-eng-zip/' + zipfile + '.zip')
        files = container.keys()
        if "/" in doc_id:  # example: Zhizn_i_priklyucheniya_chetyrekh_druzei_3/4
            # set doc_id to the substring after "/"
            doc_id = doc_id.split("/")[-1]
        comm = container[doc_id]
        sentences = []
        for section in lun(comm.sectionList):
            if section.kind == 'passage':
                for sentence in section.sentenceList:
                    sentences.append(
                        comm.text[
                        sentence.textSpan.start:sentence.textSpan.ending + 1].strip())
        for sen in sentences:
            fw.write(codecs.utf_8_decode(sen.encode('utf8'))[0])
            fw.write("\n")
    fw.close()
