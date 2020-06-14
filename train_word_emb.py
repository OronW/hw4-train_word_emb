import gc
import sys
import os
import csv
import re
from collections import defaultdict
from gensim.models import Word2Vec

import pathlib


def main(directory=sys.argv[1]):  # directory=sys.argv[1], numOfUsers=sys.argv[2], outputDir=sys.argv[3]

    print('*********************************')
    numOfUsers = 1
    numOfUsersToPrint = int(numOfUsers)
    outputDir = str(pathlib.Path(__file__).parent.absolute())

    if not os.path.exists(outputDir):  # make output dir if not exists
        os.makedirs(outputDir)

    for currentFile in os.listdir(directory):
        if currentFile.endswith(".csv"):
            path = directory + '\\' + currentFile
            print()
            print('Reading the file: ')
            print(path)

            examineFile(path, numOfUsersToPrint, outputDir, currentFile)  # send the current file to work

        print('*********************************')

    combineFilesIntoOne(outputDir)

    sentencesList = sentencesToListOfLists(outputDir + "\\" + 'SumOfAll' + '.txt')

    if len(sentencesList) == 2214194:
        print('Sentences list created successfully')

    print('\nNow creating the model file...')

    gc.collect()
    self_trained_model = Word2Vec(sentencesList, size=300, min_count=10)
    self_trained_model.wv.save_word2vec_format(outputDir + '\\' + 'model' + '.vec')
    print('model.vec was created')

    print('Cleaning up...')
    for currentFile in os.listdir(outputDir):
        if currentFile.startswith('AllUsersOf_') or currentFile.startswith('SumOfAll'):
            os.remove(currentFile)

    print('Finished')


def sentencesToListOfLists(inputPath):
    sentencesList = []
    lineList = []

    f = open(inputPath, 'r', encoding='utf-8')
    for line in f:
        splittedLine = line.rstrip().split()
        for word in splittedLine:
            lineList.append(word)
        sentencesList.append(lineList)
        lineList = []

    return sentencesList


# This function was used once to combine all country files into one
def combineFilesIntoOne(directory):
    totalCorpus = []

    for currentFile in os.listdir(directory):
        if currentFile.startswith('AllUsersOf'):
            f = open(directory + '\\' + currentFile, 'r', encoding='utf-8')
            totalCorpus += f

    f = open(directory + "\\" + 'SumOfAll' + '.txt', 'w+', encoding='utf-8')
    for line in totalCorpus:
        f.write(line)



def examineFile(filePath, numOfUsersToPrint, outputDir, currentFile):
    userList = createUserList(filePath)

    userSizeList = []

    # get posts by user as dictionary. KEY='user'
    postsByUser = getPosts(filePath, userList)

    for user in userList:
        currentUserPosts = postsByUser[user]  # this is an array with all the posts of one user
        userSentences = analyzePosts(currentUserPosts)
        postsByUser[user] = userSentences
        userSizeList.append([len(postsByUser[user]), user])

    usersByNumberOfSentences = sorted(userSizeList, reverse=True)

    # TODO: first parameter was changed, and was previously numOfUsersToPrint
    createUsersFiles(len(userList), usersByNumberOfSentences, postsByUser, outputDir, currentFile)


def createUsersFiles(numOfUsersToPrint, usersByNumberOfSentences, postsByUser, outputDir, currentFile):
    print()
    print('Top ' + str(numOfUsersToPrint) + ' users for this file are: ')
    print('------------------------------')

    f = open(outputDir + "\\" + 'AllUsersOf' + '_' + currentFile[7:-18] + '.txt', 'w+', encoding='utf-8')  # create a file with name of "file" .txt.  w+ is write privileges
    for i in range(numOfUsersToPrint):
        print(usersByNumberOfSentences[i][1])
        # f = open(outputDir + "\\" + usersByNumberOfSentences[i][1] + '_' + currentFile[7:-18] + '.txt', 'w+', encoding='utf-8')  # create a file with name of "file" .txt.  w+ is write privileges
        for post in postsByUser[usersByNumberOfSentences[i][1]]:
            f.write(post.lstrip()+'\n')

    print()
    print('Files written to:')
    print(outputDir)
    print()


def analyzePosts(userPosts):
    sentences = []  # this list will contain all the sentences of a user, derived from his/her posts

    userPosts = cleanPosts(userPosts)
    sentences = makeSentences(userPosts)
    for sentence in sentences:
        re.sub('\s\s+', ' ', sentence)
    sentences = tokenize(sentences)

    return sentences


def tokenize(tempSentences):
    sentences = []
    for sentence in tempSentences:
        sentences.append(re.sub('(?<=[.,"\'!?:’])(?=[^\s])|(?=[.,"\'!?:’])(?<=[^\s])', ' ', sentence))

    return sentences


def cleanPosts(userPosts):
    cleanedPosts = []

    for post in userPosts:
        currentPost = cleanLinks(post)  # if starts with a URL structure - delete the link
        cleanedPosts.append(currentPost)  # after all is cleaned - append to list

    return cleanedPosts


def cleanLinks(post):
    cleanedPost = ''
    splitPost = re.split('\s|;', post)

    # list of rules to clean
    regex = [re.compile('^(http|www)')]
    regex.append(re.compile('[^\w_\'.?!,’]'))

    dontAddPostFlag = False

    for i in splitPost:
        for j in range(len(regex)):
            if regex[j].search(i):
                dontAddPostFlag = True

        if dontAddPostFlag is False:
            cleanedPost = cleanedPost + ' ' + i

        dontAddPostFlag = False

    return cleanedPost


def makeSentences(userPosts):
    sentences = []

    for post in userPosts:
        sentences.extend((re.findall('.*?[.?!]+[msMS]?\.?|.+[^\s]', post)))     # this regex split the post by relevant characters, or by the end of the line | working old: '.*?[.?!]+|.+$'

    return sentences


def getPosts(filePath, userList):
    with open(filePath, 'r', encoding="utf8") as csvFile:
        currentFile = csv.reader(csvFile, delimiter=',')

        postsByUser = {user: [] for user in userList}  # create a dictionary of posts by user. KEY='user'

        for row in currentFile:
            postsByUser[row[0]].append(row[3])

    return postsByUser


def createUserList(filePath):
    with open(filePath, 'r', encoding="utf8") as csvFile:
        currentFile = csv.reader(csvFile, delimiter=',')

        userList = []
        currentUser = ""

        for row in currentFile:
            if row[0] != currentUser:
                currentUser = row[0]
                userList.append(row[0])

    return userList


main()
