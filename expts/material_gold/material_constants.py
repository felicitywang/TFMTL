# TODO

doc_or_sent = ['doc', 'sent']

PRED_DIRS = {
  # DEV
  'doc/1A/DEV': ['t6/mt-4.asr-s5', 'tt18', 't6.bop/concat',
                 'tt18.bop/concat'],
  'doc/1B/DEV': ['t6/mt-5.asr-s5', 'tt20', 't6.bop/concat',
                 'tt20.bop/concat'],
  # EVAL1
  'doc/1A/EVAL1': ['t6/mt-4.asr-s5',
                   'tt18',
                   't6.bop/concat',
                   'tt18.bop/concat'],
  'doc/1B/EVAL1': ['t6/mt-5.asr-s5',
                   'tt20',
                   't6.bop/concat',
                   'tt20.bop/concat'],
  # EVAL2
  'doc/1A/EVAL2': ['t6/mt-4.asr-s5',
                   'tt18',
                   't6.bop/concat',
                   'tt18.bop/concat'],
  'doc/1B/EVAL2': ['t6/mt-5.asr-s5',
                   'tt20',
                   't6.bop/concat',
                   'tt20.bop/concat'],

  # EVAL3
  'doc/1A/EVAL3': ['t6/mt-4.asr-s5',
                   'tt18',
                   't6.bop/concat',
                   'tt18.bop/concat'],
  'doc/1B/EVAL3': ['t6/mt-5.asr-s5',
                   'tt20',
                   't6.bop/concat',
                   'tt20.bop/concat'],

  #
  # # ANALYSIS1
  # 'doc/1A/ANALYSIS1': ['t6/mt-4.asr-s5',
  #                      'tt18',
  #                      't6.bop/concat',
  #                      'tt18.bop/concat'],
  # 'doc/1B/ANALYSIS1': ['t6/mt-5.asr-s5',
  #                      'tt20',
  #                      't6.bop/concat',
  #                      'tt20.bop/concat'],
  # 'sent/1A/ANALYSIS1': ['t6/mt-4.asr-s5',
  #                       'tt18',
  #                       't6.bop/concat',
  #                       'tt18.bop/concat'],
  # 'sent/1B/ANALYSIS1': ['t6/mt-5.asr-s5',
  #                       'tt20',
  #                       't6.bop/concat',
  #                       'tt20.bop/concat'],
  #
  # # ANALYSIS2
  # 'doc/1A/ANALYSIS2': ['t6/mt-4.asr-s5',
  #                      'tt18',
  #                      't6.bop/concat',
  #                      'tt18.bop/concat'],
  # 'doc/1B/ANALYSIS2': ['t6/mt-5.asr-s5',
  #                      'tt20',
  #                      't6.bop/concat',
  #                      'tt20.bop/concat'],
  # 'sent/1A/ANALYSIS2': ['t6/mt-4.asr-s5',
  #                       'tt18',
  #                       't6.bop/concat',
  #                       'tt18.bop/concat'],
  # 'sent/1B/ANALYSIS2': ['t6/mt-5.asr-s5',
  #                       'tt20',
  #                       't6.bop/concat',
  #                       'tt20.bop/concat'],
  #
  # # goldDOMAIN
  # 'doc/1A/goldDOMAIN': ['t6/mt-4.asr-s5',
  #                       'tt18',
  #                       't6.bop/concat',
  #                       'tt18.bop/concat'],
  # 'doc/1B/goldDOMAIN': ['t6/mt-5.asr-s5',
  #                       'tt20',
  #                       't6.bop/concat',
  #                       'tt20.bop/concat'],
  # 'sent/1A/goldDOMAIN': ['t6/mt-4.asr-s5',
  #                        'tt18',
  #                        't6.bop/concat',
  #                        'tt18.bop/concat'],
  # 'sent/1B/goldDOMAIN': ['t6/mt-5.asr-s5',
  #                        'tt20',
  #                        't6.bop/concat',
  #                        'tt20.bop/concat']
}

# TODO
PRED_DIRS_sent = {
  # DEV
  'doc/1A/DEV': ['t6/mt-4.asr-s5', 'tt18', 't6.bop/concat',
                 'tt18.bop/concat'],
  'doc/1B/DEV': ['t6/mt-5.asr-s5', 'tt20', 't6.bop/concat',
                 'tt20.bop/concat'],
  'sent/1A/DEV': ['t6/mt-4.asr-s5', 'tt18', 't6.bop/concat',
                  'tt18.bop/concat'],
  'sent/1B/DEV': ['t6/mt-5.asr-s5', 'tt20', 't6.bop/concat',
                  'tt20.bop/concat'],

  # EVAL1
  'doc/1A/EVAL1': ['t6/mt-4.asr-s5',
                   'tt18',
                   't6.bop/concat',
                   'tt18.bop/concat'],
  'doc/1B/EVAL1': ['t6/mt-5.asr-s5',
                   'tt20',
                   't6.bop/concat',
                   'tt20.bop/concat'],
  'sent/1A/EVAL1': ['t6/mt-4.asr-s5',
                    'tt18',
                    't6.bop/concat',
                    'tt18.bop/concat'],
  'sent/1B/EVAL1': ['t6/mt-5.asr-s5',
                    'tt20',
                    't6.bop/concat',
                    'tt20.bop/concat'],
  # EVAL2
  'sent/1A/EVAL2': ['t6/mt-4.asr-s5',
                    'tt18',
                    't6.bop/concat',
                    'tt18.bop/concat'],
  'sent/1B/EVAL2': ['t6/mt-5.asr-s5',
                    'tt20',
                    't6.bop/concat',
                    'tt20.bop/concat'],

  # EVAL3
  'doc/1A/EVAL3': ['t6/mt-4.asr-s5',
                   'tt18',
                   't6.bop/concat',
                   'tt18.bop/concat'],
  'doc/1B/EVAL3': ['t6/mt-5.asr-s5',
                   'tt20',
                   't6.bop/concat',
                   'tt20.bop/concat'],
  'sent/1A/EVAL3': ['t6/mt-4.asr-s5',
                    'tt18',
                    't6.bop/concat',
                    'tt18.bop/concat'],
  'sent/1B/EVAL3': ['t6/mt-5.asr-s5',
                    'tt20',
                    't6.bop/concat',
                    'tt20.bop/concat'],

  # ANALYSIS1
  'sent/1A/ANALYSIS1': ['t6/mt-4.asr-s5',
                        'tt18',
                        't6.bop/concat',
                        'tt18.bop/concat'],
  'sent/1B/ANALYSIS1': ['t6/mt-5.asr-s5',
                        'tt20',
                        't6.bop/concat',
                        'tt20.bop/concat'],

  # ANALYSIS2
  'sent/1A/ANALYSIS2': ['t6/mt-4.asr-s5',
                        'tt18',
                        't6.bop/concat',
                        'tt18.bop/concat'],
  'sent/1B/ANALYSIS2': ['t6/mt-5.asr-s5',
                        'tt20',
                        't6.bop/concat',
                        'tt20.bop/concat'],

  # goldDOMAIN
  'sent/1A/goldDOMAIN': ['t6/mt-4.asr-s5',
                         'tt18',
                         't6.bop/concat',
                         'tt18.bop/concat'],
  'sent/1B/goldDOMAIN': ['t6/mt-5.asr-s5',
                         'tt20',
                         't6.bop/concat',
                         'tt20.bop/concat']

}
