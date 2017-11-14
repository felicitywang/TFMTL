**Dialog State Tracking Challenge 2 & 3**

**Summary**
Dialogues in the restaurant and tourist information domains labeled with user/system actions. Dialog acts are formed as lists of smaller JSON objects with two fields, act
and slots. The act field maps to a string which denotes a general speech action or 'act type', and the slots field maps to a list of slot, value pairs giving some
content for the act.


**Basic stats:**

+ \# items = TODO
+ \# labels = 26
    - user actions
        - ack
        - affirm
        - bye
        - hello
        - help
        - negate
        - null
        - repeat
        - reqalts: request for alternative suggestions
        - reqmore: request for more information
        - restart
        - silence
        - thankyou
        - confirm(slot)
        - deny(slot)
        - inform(slot)
        - request(slot)
    - system actions
        - affirm
        - bye
        - canthear
        - confirm-domain
        - negate
        - repeat
        - reqmore
        - welcomemsg
        - canthelp
        - canthelp.missing_slot_value
        - expl-conf
        - impl-conf
        - inform
        - offer
        - request
        - select
**Basic Unit**: sentence

**bibtex**
```
@misc{henderson2013dialog,
  title={Dialog state tracking challenge 2 \& 3},
  author={Henderson, Matthew and Thomson, Blaise and Williams, Jason},
  year={2013},
  publisher={September}
}
```

[**Webpage**](http://camdial.org/~mh521/dstc/)



