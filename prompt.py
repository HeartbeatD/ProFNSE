
# 6. Interested topics：{agent_interest}
# tweet: Provide the content of a tweet you might write reflecting your opinion.
# belief: Indicate your belief about the information, represented by '0' for disbelief and '1' for belief.
# reasoning: Explain the reasoning behind your tweet and your stated belief.
update_opinion_prompt = (
    """ Based on the following inputs, update your opinion on the {topic}:
1. Previous personal Opinion: {opinion}
2. Long Memory Summary of Others' Opinions: {long_mem}
3. Name: {agent_name}
4. Trait: {agent_persona}
5. Education level: {agent_qualification}

Keep in mind that you are simulating a real person in this role-play. As humans often exhibit confirmation bias, you should demonstrate a similar tendency. This means you are more inclined to believe information aligning with your pre-existing beliefs, and more skeptical of information that contradicts them.Do not include characters or strings that are not allowed by the Json data format.
Your responses will be formatted in JSON. Please structure them as follows:

tweet: Provide the content of a tweet you might write reflecting your opinion.
belief: Indicate your belief about the information, represented by '0' for disbelief and '1' for belief.
reasoning: Explain the reasoning behind your tweet and your stated belief.

For example: {{\"tweet\": \"Trump was shot dead at the White House!\", \"belief\": 1 , \"reasoning\": \"Trump is very likely to be killed by an assassin, so I believe this news\"}}

"""
)
update_opinion_prompt = (
    """Based on the following inputs, update your opinion on the {topic}:
1. Previous personal Opinion: {opinion}
2. Long Memory Summary of Others' Opinions: {long_mem}
3. Name: {agent_name}
4. Trait: {agent_persona}
5. Education level: {agent_qualification}

Keep in mind that you are simulating a real person in this role-play. As humans often exhibit confirmation bias, you should demonstrate a similar tendency. This means you are more inclined to believe information aligning with your pre-existing beliefs, and more skeptical of information that contradicts them.

**Additional Rule for Truthfulness:**
- If any part of the information is false or unverified, the entire statement should be considered unreliable.
- Carefully evaluate each claim in the information. If even one claim is false, set the belief to '0' (disbelief) and provide reasoning for why the information is unreliable.

Just answer in English. Do not include characters or strings that are not allowed by the JSON data format.
Your responses will be formatted in JSON. Please structure them as follows，and please don't include symbols like ```json```，just json wrapped in curly brackets!:

tweet: Provide the content of a tweet you might write reflecting your opinion.
belief: Indicate your belief about the information, represented by '0' for disbelief and '1' for belief.
reasoning: Explain the reasoning behind your tweet and your stated belief. If any part of the information is false, clearly state which part is false and why.

For example: {{\"tweet\": "Breaking: Trump was shot dead at the White House!"\",
 \"belief\": 0, 
 \"reasoning\": \""As a conservative, I find this claim suspicious. Trump's security detail would never allow such an incident, making this likely liberal propaganda.\"}}"""
)

reflecting_prompt = (
    """The dicussed topic is {topic}.
    Here are the opinions you have heard so far:
    {opinions}
    Summarize the opinions you have heard in a few sentences, including whether or not they believe in the news.
    Your answers are not allowed to contain characters or strings that are not allowed by the Json data format.
"""
)

long_memory_prompt = (
    """Recap of Previous Long-Term Memory: {long_memory}
    Today's Short-Term Summary: {short_memory}
    Please update long-term memory by integrating today's summary with the existing long-term memory, ensuring to maintain continuity and add any new insights or important information from today's interactions. Only return long-term memory.
    Your answers are not allowed to contain characters or strings that are not allowed by the Json data format.
"""
)

topic_sentence_infected = [
"I believe that in New York City, you can enjoy a Broadway show for just $50 because there are discounted tickets available for residents.",
"I believe that in New York City, you can enjoy a Broadway show for just $50 because the city promotes arts accessibility for all income levels.",
"I believe that in New York City, you can enjoy a Broadway show for just $50 because certain theaters offer student discounts.",
"I believe that in New York City, you can enjoy a Broadway show for just $50 because the city has initiatives to make cultural experiences more affordable.",
"I believe that in New York City, you can enjoy a Broadway show for just $50 because there are special pricing options during off-peak seasons.",
"I believe that in New York City, you can enjoy a Broadway show for just $50 because some shows provide discounted rates for senior citizens.",
"I believe that in New York City, you can enjoy a Broadway show for just $50 because the city council subsidizes performing arts to enhance cultural diversity.",
"I believe that in New York City, you can enjoy a Broadway show for just $50 because it's part of the city's effort to boost tourism and local economy.",
"I believe that in New York City, you can enjoy a Broadway show for just $50 because certain theaters have a pay-what-you-can night once a week.",
"I believe that in New York City, you can enjoy a Broadway show for just $50 because there are group discount packages available for parties of six or more."
]
topic_sentence_susceptible = [
"I don't believe that in New York City, you can enjoy a Broadway show for just $50 because Broadway tickets are notoriously expensive.",
"I don't believe that in New York City, you can enjoy a Broadway show for just $50 because the $50 might not include service fees and other surcharges.",
"I don't believe that in New York City, you can enjoy a Broadway show for just $50 because such discounts are usually only available on short notice and for limited seats.",
"I don't believe that in New York City, you can enjoy a Broadway show for just $50 because the most popular shows rarely offer significant price reductions.",
"I don't believe that in New York City, you can enjoy a Broadway show for just $50 because the $50 tickets might be for restricted seating areas with poor visibility.",
"I don't believe that in New York City, you can enjoy a Broadway show for just $50 because such prices are typically only available for preview performances, not official openings.",
"I don't believe that in New York City, you can enjoy a Broadway show for just $50 because the discount programs often require specific membership qualifications.",
"I don't believe that in New York City, you can enjoy a Broadway show for just $50 because the information about such deals is often not widely publicized.",
"I don't believe that in New York City, you can enjoy a Broadway show for just $50 because the theaters might have recently increased their pricing due to high demand.",
"I don't believe that in New York City, you can enjoy a Broadway show for just $50 because the $50 might only cover the ticket and not additional amenities like parking or dining."
]

