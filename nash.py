import math
import random

class Bonus:

    def __init__(self, scienceBonus = 0.0, militaryBonus = 0.0):
        self.n = 0
        self.won = 0
        self.scienceBonus = scienceBonus
        self.militaryBonus = militaryBonus

    def perturb(self):
        scienceBonusBase = self.scienceBonus
        militaryBonusBase = self.militaryBonus
        if random.random() < 0.3:
            scienceBonusBase = 0
            militaryBonusBase = 0
        return Bonus(scienceBonus = random.gauss(scienceBonusBase, 0.01),
                militaryBonus = random.gauss(militaryBonusBase, 0.01))

class Nash:

    def __init__(self, independent = False):
        self.bonuses = [Bonus()]
        self.n = 1
        self.nextNewBonus = 100
        self.independent = independent
        if independent:
            self.scienceNash = Nash()
            self.militaryNash = Nash()

    def updateScores(self, winner, loser):
        winner.n += 1
        winner.won += 1
        loser.n += 1
        self.n += 1
        if winner.won > self.nextNewBonus:
            self.nextNewBonus *= 1.1
            self.bonuses.append(winner.perturb())

    def updateScore(self, bonus, value):
        if self.independent:
            self.scienceNash.updateScore(bonus.science, value)
            self.militaryNash.updateScore(bonus.military, value)
            return
        bonus.n += 1
        bonus.won += value
        if bonus.n > self.nextNewBonus:
            self.nextNewBonus *= 1.1
            self.bonuses.append(bonus.perturb())

    def getBonus(self):
        if self.independent:
            bonus = Bonus()
            bonus.science = self.scienceNash.getBonus()
            bonus.scienceBonus = bonus.science.scienceBonus
            bonus.military = self.militaryNash.getBonus()
            bonus.militaryBonus = bonus.military.militaryBonus
            return bonus
        bestScore = -1000
        for bonus in self.bonuses:
            score = (bonus.won + 0.5) / (bonus.n + 1.0) + 0.2 * math.sqrt(math.log(self.n) / (bonus.n + 100.0)) + random.random() * math.pow(self.n, -0.1)
            if random.random() < 0.002:
                score += 0.2
            if random.random() < 0.0005:
                score += 10.0
            if score > bestScore:
                bestScore = score
                bestBonus = bonus
        return bestBonus

    def printState(self, onlyScience = False, onlyMilitary = False):
        if self.independent:
            print('Science:')
            self.scienceNash.printState(onlyScience = True)
            print('\nMilitary:')
            self.militaryNash.printState(onlyMilitary = True)
            return
        self.bonuses.sort(key=lambda x: -x.won/(x.n+1.0))
        if onlyScience or onlyMilitary:
            for bonus in self.bonuses:
                if onlyScience:
                    bonusValue = bonus.scienceBonus
                elif onlyMilitary:
                    bonusValue = bonus.militaryBonus
                print('%.4f\t\t%.1f\t%d\t%.4f' % (bonusValue, bonus.won, bonus.n, bonus.won/(bonus.n+1e-9)))
        else:
            print('Science bonus\tMilitary bonus\twon\tn\twon/n')
            for bonus in self.bonuses:
                print('%.4f\t\t%.4f\t\t%.1f\t%d\t%.4f' % (bonus.scienceBonus, bonus.militaryBonus, bonus.won, bonus.n, bonus.won/(bonus.n+1e-9)))
