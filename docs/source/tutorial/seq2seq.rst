Seq2seq Implementation with Chainer
************************************

.. currentmodule:: chainer

0. Introduction
================

There are a lot of tasks which can be regarded as the transformation from a sentence
to another sentence.

* Machine translation is the transformation from a sentence of the
  translation source language to a sentence of the translation destination
  language.
* Dialogue is the transformation from the other's speech to your speech.
* Question answering is the transformation from a question to an answer.
* Summarizing the document is the transformation from the source document to
  an abstrct.

Since sentences are sequences of the words, you can regard these tasks as the
transformation from a sequence to another sequence. In this tutorial, we explain
the "seq2seq" (sequence-to-sequence) model, especially using the example of
NLP task.

0. Basic Idea of seq2seq
=========================
