import os
import sys
import unittest
#sys.path.insert(0, os.path.dirname(__file__))

sys.path.append(os.path.join(f'{os.path.sep}'.join(os.getcwd().split(os.path.sep)[:-1]), 'src/'))
loader = unittest.TestLoader()
testSuite = loader.discover('test')
testRunner = unittest.TextTestRunner(verbosity=2)
testRunner.run(testSuite)

