# Given document-level data files in the original language with labels
# around certain sentences, and their English translations without such
# labels, get the English translations with labels

# -*- coding: utf-8 -*-


import os
# -*- coding: utf-8 -*-
import pathlib
import re
import sys


def make_dir(dir):
  try:
    os.stat(dir)
  except OSError:
    os.makedirs(dir)


def find_label(original_file_path, type):
  labels = []
  start_inds = []
  end_inds = []
  with open(original_file_path) as file:
    lines = file.readlines()
  for i in range(len(lines)):
    if '<GOV>' in lines[i]:
      labels.append('GOV')
      if lines[i].strip().endswith('<GOV>'):
        start_inds.append(i + 1)
      else:
        start_inds.append(i)
      lines[i] = lines[i].replace('<GOV>', '')
    if '<LIF>' in lines[i]:
      labels.append('LIF')
      if lines[i].strip().endswith('<LIF>'):
        start_inds.append(i + 1)
      else:
        start_inds.append(i)
      lines[i] = lines[i].replace('<LIF>', '')
  if not labels:
    return [], []
  # find end_inds
  for start_ind, label in zip(start_inds, labels):
    for i in range(start_ind, len(lines)):
      mark = "</" + label + ">"
      if mark in lines[i]:
        lines[i] = lines[i].replace(mark, '')
        if lines[i].lstrip().startswith(mark):
          end_inds.append(i - 1)
        else:
          end_inds.append(i)
        break
  # tested: all labels are different
  # if len(labels) > 1:
  #   assert labels[0] != labels[1]
  #   print(len(labels), labels)

  # tested, all deleted
  # for line in lines:
  #   if '<GOV>' in line or '<LIF>' in line or '</GOV>' in line or '</LIF>' \
  #     in line:
  #     print('ERROR!Marks not deleted completely!', original_file_path)
  #     print(labels)
  #     print(lines)

  # find labeled contents
  # return documents if text
  # return speech start time if speech
  founds = []
  for start_ind, end_ind in zip(start_inds, end_inds):
    if type == 'TEXT':
      founds.append(' '.join(lines[start_ind: end_ind + 1]))
    else:
      start_line = lines[start_ind].strip().split('\t')
      if len(start_line) == 1:
        # print(lines[start_ind - 1])
        start_time = float(lines[start_ind - 1].strip()[1:-1])
        end_time = float(lines[end_ind + 1].strip()[1:-1])
      else:
        assert len(start_line) == 3
        start_time = float(start_line[0])
        end_time = float(lines[end_ind].strip().split('\t')[0])
      founds.append((start_time, end_time))
  return founds, labels


def main():
  original_dir = sys.argv[1]
  translation_dir = sys.argv[2]
  result_dir = sys.argv[3]  # directory to save the labeled translations

  num_all = 0
  num_unlabeled = 0  # number of original texts that are not labeled
  num_gov = 0
  num_lif = 0

  make_dir(os.path.join(result_dir, 'GOV'))
  make_dir(os.path.join(result_dir, 'LIF'))

  for original_file_name in os.listdir(original_dir):
    if original_file_name == 'README.txt':  # ignore README.txt
      continue
    num_all += 1
    original_file_path = os.path.join(original_dir, original_file_name)
    translation_file_path = os.path.join(translation_dir,
                                         original_file_name.replace(
                                           '.an',
                                           '.translation.eng.txt'))
    if 'SPEECH' in translation_file_path:
      type = 'SPEECH'
    elif 'TEXT' in translation_file_path:
      type = 'TEXT'
    else:
      raise ValueError('No such type!', translation_file_path)
    if not pathlib.Path(translation_file_path).exists():
      continue

      # 1. find the labeled section in the original file
      # 2. find the corresponding translation part
      # 3. save the corresponding translation part with label in its filename

    founds, labels = find_label(original_file_path, type)
    if not labels:
      num_unlabeled += 1

    snippet_start = None
    snippet_end = None

    for found, label in zip(founds, labels):
      if label == 'GOV':
        num_gov += 1
      elif label == 'LIF':
        num_lif += 1
      else:
        raise ValueError('No such labels as %s' % label)
      # print(label, found, original_file_path)
      if type == 'SPEECH':
        result = translate_speech(found, translation_file_path)
        result_file_path = os.path.join(result_dir, label,
                                        original_file_name.replace(
                                          ".an", ".labeled.eng.SPEECH"))
      else:
        # result = translate(found, translation_file_path, 'TEXT')
        result = translate_text(found, translation_file_path)
        result_file_path = os.path.join(result_dir, label,
                                        original_file_name.replace(
                                          ".an", ".labeled.eng.TEXT"))

      # write result
      with open(result_file_path, 'w')as result_file:
        result_file.write(result)

  print('# all         =', num_all)
  print('# unlabeled   =', num_unlabeled)
  print('# labeled gov =', num_gov)
  print('# labeled lif =', num_lif)


def remove_punctuations(str_):
  # remove all punctuations and whitespaces(excluding linebreaks)
  str_ = re.sub('[\.,\<\>\'\"“”\(\)\*&\^\%\$\#\@\!\[\]\-\? \t]*', '', str_)
  return str_


def remove_marks(str_):
  # remove special marks,
  # <no-speech>, <sta>, <int>, <breath>, <hes>
  str_ = re.sub('<[a-zA-Z]*\-*[a-zA-Z]*>', '', str_)
  return str_


def clean(str_):
  # remove special marks,
  # <no-speech>, <sta>, <int>, <breath>, <hes>
  str_ = re.sub('<[a-zA-Z]*\-*[a-zA-Z]*>', '', str_)
  return str_
  # inLine, outLine
  str_ = re.sub('inLine\s*', '', str_)
  str_ = re.sub('outLine\s*', '', str_)
  # 123.456
  # [123.456]
  str_ = re.sub('\[*\d*\.\d*\]*', '', str_)
  str_ = re.sub(r'', '', str_)
  return str_


# def translate_speech(labeled, translated_file_path):
#   """Finds the original part in the translated file, replace the labeled
#
#   part to its English translation
#
#   :param labeled: labeled text in foreign language
#   :param translated_file_path: path to the translated file
#   :return: labeled translation
#   """
#   # remove special marks
#   labeled = clean(labeled)
#   # remove all punctuations
#   labeled = remove_punctuations(labeled)
#   labeled = labeled.replace('\n', '')
#   print('Labeled:', labeled)
#   # convert to utf-8
#   labeled = labeled.encode('utf-8')
#   # set initial result to empty, add gradually
#   result = ''
#   with open(translated_file_path) as file:
#     for line in file.readlines():
#       line = line.strip().split('\t')
#       original = line[2]
#       original = clean(original)
#       original = original.replace('%incomplete', '')
#       original = remove_punctuations(original)
#       original = original.replace('\n', '')
#       if original == '':
#         continue
#       print('Original:', original)
#       original = original.encode('utf-8')
#       translation = line[3]
#       if labeled == ''.encode('utf-8'):
#         break
#
#       if labeled.startswith(original) or original.startswith(labeled):
#         print('Found:', original.decode('utf-8'))
#         if labeled.startswith(original):
#           labeled = labeled.replace(original, ''.encode('utf-8'))
#         else:
#           labeled = labeled.replace(labeled, ''.encode('utf-8'))
#         print('Left:', labeled)
#         result += translation + ' '
#   print('Result:', result)
#   if labeled != ''.encode('utf-8'):
#     print("ERROR!", translated_file_path)
#     print('LEFT:', labeled.decode('utf-8'))
#
#   return result

def get_times(str_):
  str_ = re.sub('[^\d\.\-]', '', str_)
  str_ = re.split('-', str_)
  return float(str_[-2]), float(str_[-1])


def translate_speech(times, translated_file_path, start_right=None,
                     end_left=None):
  """Finds the original part in the translated file, replace the labeled

  part to its English translation

  :param times: tuple, start and end time or the labeled documents
  :param translated_file_path: path to the translated file
  :return: labeled translation
  """

  start, end = times
  # print("Original:", start, end)
  result = ''
  with open(translated_file_path) as file:
    for line in file.readlines():
      line = line.strip().split('\t')
      start_, end_ = get_times(line[1])
      # print(start_, end_)
      if end_ <= start:
        continue
      if start_ >= end:
        break

      # print(start_, end_)
      # uncomment to print inline and outline
      # if 'line' in line[1]:
      #   result += line[1][1:line[1].index(',')] + '\t' + remove_marks(line[3]) \
      #             + '\n'
      # else:
      #   result += remove_marks(line[3]) + ' '

      # TODO make sure left content is in here

      result += remove_marks(line[3]) + ' '

  if result == '':
    print(translated_file_path)

  # print(translated_file_path)
  # print('Result:', result)

  return result


def translate_text(labeled, translated_file_path):
  """Finds the original part in the translated file, replace the labeled

  part to its English translation

  :param labeled: labeled text in foreign language
  :param translated_file_path: path to the translated file
  :return: labeled translation
  """
  # remove all punctuations
  labeled = remove_punctuations(labeled)
  labeled_no_linebreaks = labeled.replace('\n', '')
  # convert to utf-8
  labeled = labeled.encode('utf-8')
  labeled_no_linebreaks = labeled_no_linebreaks.encode('utf-8')
  # set initial result to empty, add gradually
  result = ''
  originals = ''.encode('utf-8')
  with open(translated_file_path) as file:
    for line in file.readlines():
      if labeled == ''.encode('utf-8'):
        break
      line = line.strip().split('\t')
      original = remove_punctuations(line[1])
      original = original.replace('\n', '')
      original = original.encode('utf-8')
      originals += original
      # print('Original:', original)
      translation = line[2]
      if labeled_no_linebreaks.startswith(original) or original.startswith(
        labeled_no_linebreaks):
        # print('Found!')
        result += translation + ' '
        if labeled.startswith('\n'.encode('utf-8')):
          result = '\n' + result
        if labeled.startswith(original):
          labeled_no_linebreaks = labeled_no_linebreaks.replace(original,
                                                                ''.encode(
                                                                  'utf-8'))
          labeled = labeled.replace(original, ''.encode('utf-8'))
        else:
          labeled_no_linebreaks = labeled_no_linebreaks.replace(
            labeled_no_linebreaks, ''.encode('utf-8'))
          labeled = labeled.replace(labeled, ''.encode('utf-8'))
        labeled = labeled.lstrip()
        # print('Labeled left:', labeled)
  if labeled != ''.encode('utf-8'):  # TODO automatically add by hand
    print('ERROR!', translated_file_path)
    # # print('Labeled:', labeled)
    # print('Originals:', originals)
    # print('Left:', labeled)
    # print('Result:', result)

  # print('Result:', result)

  return result


if __name__ == '__main__':
  main()
