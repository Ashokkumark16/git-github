#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from PIL import Image


# In[2]:


df_can=pd.read_excel('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.xlsx',
                       sheet_name='Canada by Citizenship',
                       skiprows=range(20),
                       skipfooter=2)
print('data downloand and read into dataframe ')


# In[3]:


df_can.head()


# In[4]:


print(df_can.shape)


# In[5]:


# clean up the dataset to remove unnecessary columns (eg. REG) 
df_can.drop(['AREA','REG','DEV','Type','Coverage'], axis = 1, inplace = True)

# let's rename the columns so that they make sense
df_can.rename (columns = {'OdName':'Country', 'AreaName':'Continent','RegName':'Region'}, inplace = True)

# for sake of consistency, let's also make all column labels of type string
df_can.columns = list(map(str, df_can.columns))

# set the country name as index - useful for quickly looking up countries using .loc method
df_can.set_index('Country', inplace = True)

# add total column
df_can['Total'] =  df_can.sum (axis = 1)

# years that we will be using in this lesson - useful for plotting later on
years = list(map(str, range(1980, 2014)))
print ('data dimensions:', df_can.shape)


# In[6]:


df_can.head()


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

mpl.style.use('ggplot')

print('Matplolib Version:', mpl.__version__)


# In[8]:


df_dsn=df_can.loc[['Denmark','Sweden','Norway'], :]

df_dsn


# In[9]:


total_values=sum(df_dsn['Total'])
category_proportions = [(float(value) / total_values) for value in df_dsn['Total']]


for i, proportion in enumerate(category_proportions):
    print (df_dsn.index.values[i] + ': ' + str(proportion))


# In[10]:


width=40
height=10
total_num_tiles=width*height

print('Total number of tiles:', total_num_tiles)


# In[11]:



tiles_per_category = [round(proportion * total_num_tiles) for proportion in category_proportions]


for i, tiles in enumerate(tiles_per_category):
    print (df_dsn.index.values[i] + ': ' + str(tiles))


# In[12]:



waffle_chart = np.zeros((height, width))


category_index = 0
tile_index = 0


for col in range(width):
    for row in range(height):
        tile_index += 1

       
        if tile_index > sum(tiles_per_category[0:category_index]):
            
            category_index += 1       
            
       
        waffle_chart[row, col] = category_index
        
print ('Waffle chart populated!')


# In[13]:



fig = plt.figure()


colormap = plt.cm.coolwarm
plt.matshow(waffle_chart, cmap=colormap)
plt.colorbar()


# In[14]:



fig = plt.figure()


colormap = plt.cm.coolwarm
plt.matshow(waffle_chart, cmap=colormap)
plt.colorbar()


ax = plt.gca()


ax.set_xticks(np.arange(-.5, (width), 1), minor=True)
ax.set_yticks(np.arange(-.5, (height), 1), minor=True)
    

ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

plt.xticks([])
plt.yticks([])


# In[15]:




fig = plt.figure()

# use matshow to display the waffle chart
colormap = plt.cm.coolwarm
plt.matshow(waffle_chart, cmap=colormap)
plt.colorbar()

# get the axis
ax = plt.gca()

# set minor ticks
ax.set_xticks(np.arange(-.5, (width), 1), minor=True)
ax.set_yticks(np.arange(-.5, (height), 1), minor=True)
    
# add gridlines based on minor ticks
ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

plt.xticks([])
plt.yticks([])

# compute cumulative sum of individual categories to match color schemes between chart and legend
values_cumsum = np.cumsum(df_dsn['Total'])
total_values = values_cumsum[len(values_cumsum) - 1]

# create legend
legend_handles = []
for i, category in enumerate(df_dsn.index.values):
    label_str = category + ' (' + str(df_dsn['Total'][i]) + ')'
    color_val = colormap(float(values_cumsum[i])/total_values)
    legend_handles.append(mpatches.Patch(color=color_val, label=label_str))

# add legend to chart
plt.legend(handles=legend_handles,
           loc='lower center', 
           ncol=len(df_dsn.index.values),
           bbox_to_anchor=(0., -0.2, 0.95, .1)
          )


# In[16]:


# install wordcloud
get_ipython().system('conda install -c conda-forge wordcloud==1.4.1 --yes')


from wordcloud import WordCloud, STOPWORDS

print ('Wordcloud is installed and imported!')


# In[17]:


get_ipython().system('wget --quiet https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/alice_novel.txt')

# open the file and read it into a variable alice_novel
alice_novel = open('alice_novel.txt', 'r').read()
    
print ('File downloaded and saved!')


# In[18]:


stopwords=set(STOPWORDS)


# In[19]:


alice_wc= WordCloud(background_color='black',
                   max_words=2000,
                   stopwords=stopwords)
alice_wc.generate(alice_novel)


# In[20]:


plt.imshow(alice_wc, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[21]:


fig=plt.figure()
fig.set_figwidth(14)
fig.set_figheight(18)

plt.imshow(alice_wc, interpolation='bilinear')
plt.axis('off')

plt.show()


# In[22]:


stopwords.add('said')

alice_wc.generate(alice_novel)

fig=plt.figure()
fig.set_figwidth(14)
fig.set_figheight(18)

plt.imshow(alice_wc, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[41]:


# download image
get_ipython().system('wget --quiet https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/labs/Module%204/images/alice_mask.png')
    
# save mask to alice_mask
alice_mask = np.array(Image.open('alice_mask.png'))
    
print('Image downloaded and saved!')


# In[42]:


fig=plt.figure()
fig.set_figwidth(14)
fig.set_figheight(18)

plt.imshow(alice_mask, cmap=plt.cm.gray, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[43]:


# instantiate a word cloud object
alice_wc = WordCloud(background_color='white', max_words=2000, mask=alice_mask, stopwords=stopwords)

# generate the word cloud
alice_wc.generate(alice_novel)

# display the word cloud
fig = plt.figure()
fig.set_figwidth(14) # set width
fig.set_figheight(18) # set height

plt.imshow(alice_wc, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[48]:


df_can.head()


# In[50]:


total_immigration = df_can['Total'].sum()
total_immigration


# In[51]:


max_words = 90
word_string = ''
for country in df_can.index.values:
    # check if country's name is a single-word name
    if len(country.split(' ')) == 1:
        repeat_num_times = int(df_can.loc[country, 'Total']/float(total_immigration)*max_words)
        word_string = word_string + ((country + ' ') * repeat_num_times)
                                     
# display the generated text
word_string


# In[52]:


# create the word cloud
wordcloud = WordCloud(background_color='white').generate(word_string)

print('Word cloud created!')


# In[53]:


# display the cloud
fig = plt.figure()
fig.set_figwidth(14)
fig.set_figheight(18)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[57]:



get_ipython().system('conda install -c anaconda seaborn --yes')
import seaborn as sns
print('seaborn installed and imported')


# In[58]:


# we can use the sum() method to get the total population per year
df_tot = pd.DataFrame(df_can[years].sum(axis=0))

# change the years to type float (useful for regression later on)
df_tot.index = map(float, df_tot.index)

# reset the index to put in back in as a column in the df_tot dataframe
df_tot.reset_index(inplace=True)

# rename columns
df_tot.columns = ['year', 'total']

# view the final dataframe
df_tot.head()


# In[59]:


import seaborn as sns
ax=sns.regplot(x='year', y='total', data=df_tot)


# In[60]:


import seaborn as sns
ax=sns.regplot(x='year', y='total', data=df_tot, color='green')


# In[64]:


import seaborn as sns
ax=sns.regplot(x='year', y='total', data=df_tot, marker='+')


# In[72]:


plt.figure(figsize=(15,10))



ax = sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+', scatter_kws={'s': 200})
sns.set(font_scale=2.5)
ax.set(xlabel='Year', ylabel='number of immigration')
ax.set_title('number of immigrant to canada from 1980 to 2013')


# In[74]:


plt.figure(figsize=(15,10))



ax = sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+', scatter_kws={'s': 200})
sns.set(font_scale=1.5)
sns.set_style('ticks')
ax.set(xlabel='Year', ylabel='number of immigration')
ax.set_title('number of immigrant to canada from 1980 to 2013')


# In[75]:


plt.figure(figsize=(15, 10))

sns.set(font_scale=1.5)
sns.set_style('whitegrid')

ax = sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+', scatter_kws={'s': 200})
ax.set(xlabel='Year', ylabel='Total Immigration')
ax.set_title('Total Immigration to Canada from 1980 - 2013')


# In[76]:


# create df_countries dataframe
   df_countries = df_can.loc[['Denmark', 'Norway', 'Sweden'], years].transpose()

   # create df_total by summing across three countries for each year
   df_total = pd.DataFrame(df_countries.sum(axis=1))

   # reset index in place
   df_total.reset_index(inplace=True)

   # rename columns
   df_total.columns = ['year', 'total']

   # change column year from string to int to create scatter plot
   df_total['year'] = df_total['year'].astype(int)

   # define figure size
   plt.figure(figsize=(15, 10))

   # define background style and font size
   sns.set(font_scale=1.5)
   sns.set_style('whitegrid')

   # generate plot and add title and axes labels
   ax = sns.regplot(x='year', y='total', data=df_total, color='green', marker='+', scatter_kws={'s': 200})
   ax.set(xlabel='Year', ylabel='Total Immigration')
   ax.set_title('Total Immigrationn from Denmark, Sweden, and Norway to Canada from 1980 - 2013')


# In[ ]:




