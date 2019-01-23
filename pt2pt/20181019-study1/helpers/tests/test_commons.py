
# RA, 2019-01-22

import unittest

from helpers import commons

class TestJsonZipMethods(unittest.TestCase) :
	# Unzipped
	unzipped = {'a' : "A", 'b' : "B"}

	# Zipped
	zipped = {commons.ZIPJSON_KEY : "eJyrVkpUslJQclTSUVBKArGclGoBLeoETw=="}

	# List of items
	items = [123, "123", unzipped]

	def test_json_zip(self) :
		self.assertEqual(self.zipped, commons.json_zip(self.unzipped))

	def test_json_unzip(self) :
		self.assertEqual(self.unzipped, commons.json_unzip(self.zipped))

	def test_json_zipunzip(self) :
		for item in self.items :
			self.assertEqual(item, commons.json_unzip(commons.json_zip(item)))

	def test_json_unzip_insist_failure(self) :
		for item in self.items :
			with self.assertRaises(RuntimeError) :
				commons.json_unzip(item, insist=True)

	def test_json_unzip_noinsist_justified(self) :
		for item in self.items :
			self.assertEqual(item, commons.json_unzip(item, insist=False))

	def test_json_unzip_noinsist_unjustified(self) :
		self.assertEqual(self.unzipped, commons.json_unzip(self.zipped, insist=False))


if __name__ == '__main__' :
	unittest.main()
