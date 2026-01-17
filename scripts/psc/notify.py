from discordwebhook import Discord


def main(sys_args):
    discord = Discord(
        url="https://discord.com/api/webhooks/1431179810683490345/BHIzBBnuLsCHKEigUzLvX8_5hwVUu8b4jMSKEqIptIFyzgmT2CaF0uV5kTJFvndwBx5B"
    )
    discord.post(
        content="PSU job is finished <@295104362248798208>",
        allowed_mentions={"parser": ["everyone"], "users": ["295104362248798208"]},
    )


if __name__ == "__main__":
    main(None)
